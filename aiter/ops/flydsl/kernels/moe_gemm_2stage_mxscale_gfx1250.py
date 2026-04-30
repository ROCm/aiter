# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors


"""gfx1250 MoE 2-stage mxscale kernels (fp4/fp8/a8w4).

Implements stage1/stage2 single-kernel inline paths using the
``wmma_scale_f32_16x16x128_f8f6f4`` and ``wmma_scale_f32_32x16x128_f4``
instructions for microscaling block formats with E8M0 scales.
"""

from __future__ import annotations

import functools

from flydsl.runtime.device import get_rocm_arch as get_hip_arch

from kernels.moe_gemm_2stage import (
    MoeGemm2Mode,
    compile_moe_reduction,
)
from kernels.moe_gemm_2stage_common_gfx1250 import (
    _Stage1GateUpPackedWrapper,
    _compute_mxscale_tiling,
    _compute_pipeline_plan,
    _compute_tdm_store_layout,
    _emit_stage1_gate_up_epilogue,
    _emit_stage1_gate_up_splitk_epilogue,
    _emit_stage2_store_epilogue,
    _extract_sub8,
    _finalize_alloc_and_launch_2d,
    _make_moe_wave_layout,
    _make_mxscale_data_loaders,
    _make_wmma_sub_tiles,
    _moe_out_elem_ty,
    _mxscale_emit_wmma,
    _pick_mxscale_launch_shape,
    _require_gfx1250,
)

@functools.lru_cache(maxsize=64)
def _compile_stage1_mxscale_kernel_impl(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    route_tile_m: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    m_warp: int,
    n_warp: int,
    doweight_stage1: bool,
    out_dtype: str,
    waves_per_eu: int | None,
    data_format: str = "fp8",
    expert_sched_mode: bool = True,
    num_buffers: int = 1,
    use_tdm_gather: bool = True,
    use_tdm_gather_as: bool = True,
    use_tdm_store: bool = False,
    inst_prefetch: bool = False,
    wave_specialized_tdm: bool = False,
    cluster_m: int = 1,
    cluster_n: int = 1,
    k_batch: int = 1,
):
    """Compile mxscale stage1 single kernel (route-pack + TDM + WMMA_SCALE + epilog).

    ``use_tdm_gather_as`` enables the TDM-gather path for the A-scale matrix,
    moving that load off ``ds_cnt`` and onto ``tdm_cnt`` to eliminate the
    ``s_wait_dscnt`` stalls that dominate the scalar per-byte fallback. Falls
    back to the vectorised scalar path when the LDS scale layout is not
    row-major (``wmma_m_rep > 1`` and not ``is_fp4``) or the row width is below
    the TDM gather minimum (``scale_k_per_tile < 4``).
    """
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import llvm as llvm_dialect
    from flydsl._mlir.dialects import memref, scf
    from flydsl.compiler.kernel_function import CompilationContext
    from flydsl.expr import arith, buffer_ops, const_expr, gpu, idx2crd, range_constexpr, rocdl, tdm_ops, vector
    from flydsl.expr.typing import T
    from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr, get_op_result_or_value

    tp = _compute_mxscale_tiling(
        data_format=data_format, K=int(model_dim),
        tile_m=int(tile_m), tile_n=int(tile_n), tile_k=int(tile_k),
        m_warp=int(m_warp), n_warp=int(n_warp), out_dtype=out_dtype,
        num_buffers=int(num_buffers), cluster_m=int(cluster_m),
        cluster_n=int(cluster_n), stage_name="stage1",
    )
    is_fp4, is_a8w4 = tp["is_fp4"], tp["is_a8w4"]
    PACK_FACTOR_A, PACK_FACTOR_B = tp["PACK_FACTOR_A"], tp["PACK_FACTOR_B"]
    ACC_VEC_SIZE = tp["ACC_VEC_SIZE"]
    DS_LOADS_PER_A_FRAG = tp["DS_LOADS_PER_A_FRAG"]
    WMMA_M, WMMA_N, WMMA_K = tp["WMMA_M"], tp["WMMA_N"], tp["WMMA_K"]
    SCALE_BLOCK, SCALES_PER_WMMA = tp["SCALE_BLOCK"], tp["SCALES_PER_WMMA"]
    WAVE_SIZE = tp["WAVE_SIZE"]
    LDS_PAD_A_BYTES, LDS_PAD_B_BYTES = tp["LDS_PAD_A_BYTES"], tp["LDS_PAD_B_BYTES"]
    use_cluster = tp["use_cluster"]
    K = tp["K"]
    K_packed_a, K_packed_b = tp["K_packed_a"], tp["K_packed_b"]
    packed_tile_k_a, packed_tile_k_b = tp["packed_tile_k_a"], tp["packed_tile_k_b"]
    K_scale, scale_k_per_tile = tp["K_scale"], tp["scale_k_per_tile"]
    block_threads = tp["block_threads"]
    warp_tile_m, warp_tile_n = tp["warp_tile_m"], tp["warp_tile_n"]
    wmma_m_rep, wmma_n_rep = tp["wmma_m_rep"], tp["wmma_n_rep"]
    k_wmma_steps, n_accs = tp["k_wmma_steps"], tp["n_accs"]
    num_k_tiles = tp["num_k_tiles"]
    b_scale_load_rep = tp["b_scale_load_rep"]
    interleaved_scale_cols_b = tp["interleaved_scale_cols_b"]
    lds_a_stride_bytes = tp["lds_a_stride_bytes"]
    lds_b_stride_bytes = tp["lds_b_stride_bytes"]
    lds_a_data_bytes, lds_b_data_bytes = tp["lds_a_data_bytes"], tp["lds_b_data_bytes"]
    lds_a_scale_bytes, lds_b_scale_bytes = tp["lds_a_scale_bytes"], tp["lds_b_scale_bytes"]
    interleaved_scale_cols_a = tp["interleaved_scale_cols_a"]

    N = int(inter_dim)

    # ── Split-K validation / setup ────────────────────────────────────
    # When k_batch > 1 the K dimension (model_dim) is split across the
    # grid z-dim. Each CTA computes a K-slice and atomically accumulates
    # gate / up partial sums into a [tokens*topk, 2*inter_dim] output.
    # silu/mul fusion, doweight_stage1 and TDM store must be disabled for
    # split-K; a separate reduction kernel fuses silu*mul and folds in
    # the per-slot routing weight.
    _is_splitk = int(k_batch) > 1
    if _is_splitk:
        if int(model_dim) % int(k_batch) != 0:
            raise ValueError(
                f"split-K requires model_dim divisible by k_batch, "
                f"got model_dim={model_dim}, k_batch={k_batch}")
        _k_per_batch = int(model_dim) // int(k_batch)
        if _k_per_batch % int(tile_k) != 0:
            raise ValueError(
                f"split-K requires (model_dim // k_batch) divisible by tile_k, "
                f"got k_per_batch={_k_per_batch}, tile_k={tile_k}")
        if bool(use_tdm_store):
            raise ValueError("split-K stage1 does not support use_tdm_store")
        if bool(wave_specialized_tdm):
            raise ValueError("split-K stage1 does not support wave_specialized_tdm")
        if bool(doweight_stage1):
            raise ValueError(
                "split-K stage1 does not support fused doweight_stage1; "
                "apply routing weight in the external reduction kernel")
        _s1_out = str(out_dtype).strip().lower()
        if _s1_out not in ("f16", "fp16", "half", "bf16", "bfloat16"):
            raise ValueError(
                f"split-K stage1 only supports fp16/bf16 output (x2 atomic fadd), "
                f"got out_dtype={out_dtype!r}")
        num_k_tiles_per_bz = _k_per_batch // int(tile_k)
    else:
        _k_per_batch = int(model_dim)
        num_k_tiles_per_bz = num_k_tiles

    _merge_gate_up_tdm = bool((data_format in ("fp8", "a8w4")) and (N % int(tile_n) == 0))
    num_warps_s1 = int(m_warp) * int(n_warp)
    _tdm_loader_waves = 2 if _merge_gate_up_tdm else 4
    if bool(wave_specialized_tdm):
        if num_warps_s1 < _tdm_loader_waves:
            raise ValueError(
                f"wave_specialized_tdm requires at least {_tdm_loader_waves} waves, got {num_warps_s1}")
    tdm_desc_num_warps = 1 if bool(wave_specialized_tdm) else num_warps_s1
    effective_waves_per_eu = waves_per_eu
    if use_cluster and effective_waves_per_eu is None:
        effective_waves_per_eu = 2

    _sub_tiles = _make_wmma_sub_tiles(
        wmma_m_rep=wmma_m_rep, wmma_n_rep=wmma_n_rep, WMMA_M=WMMA_M, is_fp4=is_fp4
    )

    # A-scale TDM gather gating: requires A-side TDM gather (for _a_tok_ids
    # SGPR caches), a row-major LDS scale layout (fp4 path is always row-major;
    # non-fp4 is row-major only when wmma_m_rep == 1), and a gather row width
    # of at least 4 bytes (TDM gather hardware constraint: row_width * elem_bytes % 4 == 0 and > 0).
    _as_layout_rowmajor = bool(is_fp4) or (int(wmma_m_rep) == 1)
    _as_row_bytes_ok = int(scale_k_per_tile) >= 4 and (int(scale_k_per_tile) % 4 == 0)
    _use_tdm_gather_as = (
        bool(use_tdm_gather_as)
        and bool(use_tdm_gather)
        and _as_layout_rowmajor
        and _as_row_bytes_ok
    )

    # Pipeline calculations for multi-buffer
    _use_pipeline = int(num_buffers) >= 2
    if _use_pipeline:
        from kernels.gemm_common_gfx1250 import (
            pipeline_fence, pipeline_fence_signal, pipeline_fence_wait,
        )
        if _merge_gate_up_tdm:
            _B_TDM_PER_STEP = 1 if bool(wave_specialized_tdm) else 2
        else:
            _B_TDM_PER_STEP = 1 if bool(wave_specialized_tdm) else 4
        _pp = _compute_pipeline_plan(
            num_k_tiles=num_k_tiles_per_bz, num_buffers=int(num_buffers),
            B_TDM_PER_STEP=_B_TDM_PER_STEP, tile_m=int(tile_m),
            use_tdm_gather=use_tdm_gather,
            use_tdm_gather_as=_use_tdm_gather_as,
            wave_specialized_tdm=wave_specialized_tdm,
            tdm_loader_waves=_tdm_loader_waves,
        )
        pre_loaded = _pp["pre_loaded"]
        loop_iters = _pp["loop_iters"]
        _tail_start = _pp["tail_start"]
        extra = _pp["extra"]
        _A_GATHER_GROUPS = _pp["A_GATHER_GROUPS"]
        _AS_GATHER_GROUPS = _pp["AS_GATHER_GROUPS"]
        TDM_PER_STEP = _pp["TDM_PER_STEP"]
        _fence_outstanding = _pp["fence_outstanding"]
        _tail_plan = _pp["tail_plan"]
    from kernels.gemm_common_gfx1250 import workgroup_barrier

    alloc = SmemAllocator(
        None,
        arch=str(get_hip_arch()),
        global_sym_name=(
            f"moe_mxscale_{data_format}_s1_single_g{int(bool(use_tdm_gather))}"
            f"_as{int(_use_tdm_gather_as)}"
        ),
    )
    _nb = int(num_buffers)
    off_ag_list, off_as_list = [], []
    off_bg_list, off_bs_list = [], []
    off_bu_list, off_bsu_list = [], []
    off_bg_pair_list, off_bs_pair_list = [], []
    for _buf_i in range(_nb):
        _o = alloc._align(alloc.ptr, 16); alloc.ptr = _o + lds_a_data_bytes; off_ag_list.append(_o)
        if _merge_gate_up_tdm:
            _o = alloc._align(alloc.ptr, 16); alloc.ptr = _o + 2 * lds_b_data_bytes; off_bg_pair_list.append(_o)
        else:
            _o = alloc._align(alloc.ptr, 16); alloc.ptr = _o + lds_b_data_bytes; off_bg_list.append(_o)
        _o = alloc._align(alloc.ptr, 16); alloc.ptr = _o + lds_a_scale_bytes; off_as_list.append(_o)
        if _merge_gate_up_tdm:
            _o = alloc._align(alloc.ptr, 16); alloc.ptr = _o + 2 * lds_b_scale_bytes; off_bs_pair_list.append(_o)
        else:
            _o = alloc._align(alloc.ptr, 16); alloc.ptr = _o + lds_b_scale_bytes; off_bs_list.append(_o)
            _o = alloc._align(alloc.ptr, 16); alloc.ptr = _o + lds_b_data_bytes; off_bu_list.append(_o)
            _o = alloc._align(alloc.ptr, 16); alloc.ptr = _o + lds_b_scale_bytes; off_bsu_list.append(_o)

    # lds_tid: preloaded sorted_token_ids for current M-tile (tile_m entries, i32).
    # Used to replace per-thread buffer_load(sorted_rsrc, ...) in the K-loop A-data/A-scale
    # loaders and the epilogue. Invalid rows are pre-filled with sentinel 0xFFFFFFFF so
    # that downstream tok/slot checks naturally reject them without needing row_in_route
    # or row_in_valid masks.
    lds_tid_bytes = int(tile_m) * 4
    off_tid = alloc._align(alloc.ptr, 16)
    alloc.ptr = off_tid + lds_tid_bytes

    if bool(use_tdm_store):
        from kernels.gemm_common_gfx1250 import store_acc_vec8_to_lds
        _ds1 = _compute_tdm_store_layout(
            warp_tile_m=warp_tile_m, warp_tile_n=warp_tile_n,
            num_warps=num_warps_s1, WMMA_N=WMMA_N, use_pipeline=_use_pipeline,
        )
        total_d_bytes_s1 = _ds1["total_d_bytes"]
        lds_d_row_stride_s1 = _ds1["lds_d_row_stride"]
        warp_d_bytes_s1 = _ds1["warp_d_bytes"]
        d_output_off_s1 = _ds1["d_output_off"]
        _lds_d_stride_elems_s1 = _ds1["lds_d_stride_elems"]
        _warp_d_elems_s1 = _ds1["warp_d_elems"]
        _n_col_d_elems_s1 = _ds1["n_col_d_elems"]
        d_need_epilogue_fence_s1 = _ds1["d_need_epilogue_fence"]
        elem_bytes_d_s1 = 2
        LDS_PAD_D_BYTES_s1 = 16
        if total_d_bytes_s1 > alloc.ptr:
            alloc.ptr = total_d_bytes_s1

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def moe_mxscale_stage1_single(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_num_valid_ids: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_inter_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
    ):
        _ = i32_k_in
        # ASTRewriter strips ``const_expr(...)`` from ``if`` tests, which would
        # otherwise eliminate every reference to ``const_expr`` from the
        # rewritten function body and shrink ``co_freevars`` by one — causing
        # CPython to reject ``f.__code__ = new_f_code_o`` because the original
        # ``__closure__`` length no longer matches. Keep one explicit reference
        # so the rewritten code object's free-vars list still includes
        # ``const_expr``.
        _keep_const_expr_ref = const_expr  # noqa: F841
        if const_expr(inst_prefetch):
            if arith.cmpi(arith.CmpIPredicate.eq, rocdl.wave_id(),
                          arith.constant(0, type=T.i32)):
                _prefetch_lines = ["s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 8, 1), 1"]
                for _pg in range_constexpr(10):
                    _prefetch_lines.append(
                        f"s_prefetch_inst_pc_rel {_pg * 4096}, s0, 31")
                llvm_dialect.inline_asm(
                    None, [],
                    "\n".join(_prefetch_lines),
                    "", has_side_effects=True,
                )
        llvm_dialect.inline_asm(
            None, [],
            "s_setreg_imm32_b32 hwreg(26, 4, 1), 1",
            "",
            has_side_effects=True,
        )

        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        by = gpu.block_id("y")

        # Split-K: bz identifies the K-slice; k_base_idx is the starting
        # K offset (in data elements, pre-pack) for this CTA.
        if _is_splitk:
            bz = gpu.block_id("z")  # already index type
            k_base_idx = bz * arith.index(int(_k_per_batch))
        else:
            k_base_idx = arith.index(0)

        tokens_idx = arith.index_cast(T.index, i32_tokens_in)
        size_expert_ids = arith.index_cast(T.index, i32_size_expert_ids_in)
        c_topk_i32 = arith.constant(int(topk), type=T.i32)
        num_valid_i32 = buffer_ops.buffer_load(
            buffer_ops.create_buffer_resource(arg_num_valid_ids, max_size=True),
            arith.constant(0, type=T.i32),
            vec_width=1,
            dtype=T.i32,
        )
        sorted_num = size_expert_ids * arith.index(int(route_tile_m))
        sorted_nbytes = sorted_num * arith.index(4)
        eid_nbytes = size_expert_ids * arith.index(4)
        x_nbytes = tokens_idx * arith.index(K_packed_a)
        sx_nbytes = tokens_idx * arith.index(K_scale)
        w_rows = arith.index(int(experts * (2 * N)))
        w_nbytes = w_rows * arith.index(K_packed_b)
        sw_nbytes = w_rows * arith.index(K_scale)

        sorted_rsrc = buffer_ops.create_buffer_resource(arg_sorted_token_ids, max_size=False, num_records_bytes=sorted_nbytes)
        eid_rsrc = buffer_ops.create_buffer_resource(arg_expert_ids, max_size=False, num_records_bytes=eid_nbytes)
        x_rsrc = buffer_ops.create_buffer_resource(arg_x, max_size=False, num_records_bytes=x_nbytes)
        sx_rsrc = buffer_ops.create_buffer_resource(arg_scale_x, max_size=False, num_records_bytes=sx_nbytes)
        w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False, num_records_bytes=w_nbytes)
        sw_rsrc = buffer_ops.create_buffer_resource(arg_scale_w, max_size=False, num_records_bytes=sw_nbytes)
        out_rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=True)
        tw_rsrc = buffer_ops.create_buffer_resource(arg_sorted_weights, max_size=True)

        eid_i32 = buffer_ops.buffer_load(eid_rsrc, arith.index_cast(T.i32, by), vec_width=1, dtype=T.i32)
        eid_ok0 = arith.cmpi(arith.CmpIPredicate.sge, eid_i32, arith.constant(0, type=T.i32))
        eid_ok1 = arith.cmpi(arith.CmpIPredicate.slt, eid_i32, arith.constant(int(experts), type=T.i32))
        block_row_start = arith.index_cast(T.i32, by * arith.index(int(route_tile_m)))
        block_in_valid = arith.cmpi(arith.CmpIPredicate.slt, block_row_start, num_valid_i32)
        block_ok = arith.andi(block_in_valid, arith.andi(eid_ok0, eid_ok1))

        layout_thr = _make_moe_wave_layout(m_warp=m_warp, n_warp=n_warp, WAVE_SIZE=WAVE_SIZE, fx=fx)
        thr_coord = idx2crd(tx, layout_thr)
        wave_m_idx, wave_n_idx, lane_kgrp, lane16 = (
            fx.get(thr_coord, 0), fx.get(thr_coord, 1), fx.get(thr_coord, 2), fx.get(thr_coord, 3)
        )
        warp_m_base = wave_m_idx * arith.index(warp_tile_m)
        warp_n_base = wave_n_idx * arith.index(warp_tile_n)
        blk_n = bx * arith.index(int(tile_n))

        if const_expr(use_cluster):
            _local_x, _local_y = gpu.compute_cluster_position()
            _a_mcast_mask, b_mcast_mask = gpu.compute_mcast_masks(
                _local_x, _local_y, int(cluster_m), int(cluster_n))
        else:
            b_mcast_mask = 0

        base_ptr = alloc.get_base()
        lds_ag_bufs, lds_as_bufs = [], []
        lds_bg_bufs, lds_bs_bufs = [], []
        lds_bu_bufs, lds_bsu_bufs = [], []
        lds_bg_pair_bufs, lds_bs_pair_bufs = [], []
        for _bi in range_constexpr(_nb):
            lds_ag_bufs.append(get_op_result_or_value(
                SmemPtr(base_ptr, off_ag_list[_bi], T.i8, shape=(lds_a_data_bytes,)).get()))
            lds_as_bufs.append(get_op_result_or_value(
                SmemPtr(base_ptr, off_as_list[_bi], T.i8, shape=(lds_a_scale_bytes,)).get()))
            if const_expr(_merge_gate_up_tdm):
                lds_bg_pair_bufs.append(get_op_result_or_value(
                    SmemPtr(base_ptr, off_bg_pair_list[_bi], T.i8, shape=(2 * lds_b_data_bytes,)).get()))
                lds_bs_pair_bufs.append(get_op_result_or_value(
                    SmemPtr(base_ptr, off_bs_pair_list[_bi], T.i8, shape=(2 * lds_b_scale_bytes,)).get()))
            else:
                lds_bg_bufs.append(get_op_result_or_value(
                    SmemPtr(base_ptr, off_bg_list[_bi], T.i8, shape=(lds_b_data_bytes,)).get()))
                lds_bs_bufs.append(get_op_result_or_value(
                    SmemPtr(base_ptr, off_bs_list[_bi], T.i8, shape=(lds_b_scale_bytes,)).get()))
                lds_bu_bufs.append(get_op_result_or_value(
                    SmemPtr(base_ptr, off_bu_list[_bi], T.i8, shape=(lds_b_data_bytes,)).get()))
                lds_bsu_bufs.append(get_op_result_or_value(
                    SmemPtr(base_ptr, off_bsu_list[_bi], T.i8, shape=(lds_b_scale_bytes,)).get()))

        lds_tid = SmemPtr(base_ptr, off_tid, T.i32, shape=(int(tile_m),)).get()

        if const_expr(bool(use_tdm_store)):
            from kernels.gemm_common_gfx1250 import get_lds_memref
            d_lds_f16_count_s1 = total_d_bytes_s1 // 2
            d_smem_s1 = SmemPtr(base_ptr, d_output_off_s1, T.f16,
                                shape=(d_lds_f16_count_s1,))
            d_lds_buffer_s1 = get_lds_memref(d_smem_s1)
            warp_lds_off_s1 = (
                (wave_m_idx * arith.index(int(n_warp)) + wave_n_idx)
                * arith.index(_warp_d_elems_s1)
            )
            d_lane_base_s1 = (
                warp_lds_off_s1
                + lane16 * arith.index(_lds_d_stride_elems_s1)
                + lane_kgrp * arith.index(4 * elem_bytes_d_s1)
            )
            wave_id_idx_s1 = arith.index_cast(T.index, rocdl.wave_id())
            d_warp_off_sgpr_s1 = (
                wave_id_idx_s1 * arith.index(warp_d_bytes_s1)
                + arith.index(d_output_off_s1)
            )
            warp_m_off_sgpr_s1 = (
                (wave_id_idx_s1 / arith.index(int(n_warp)))
                * arith.index(warp_tile_m)
            )
            warp_n_off_sgpr_s1 = (
                (wave_id_idx_s1 % arith.index(int(n_warp)))
                * arith.index(warp_tile_n)
            )
            # TDM store for MoE stage1 uses gather-store mode because the
            # output rows are not contiguous — each sorted row maps to
            # out[tok * topk + slot, :] which is a scattered layout.
            # d_desc_s1 is built lazily in the epilogue after sorted_ids
            # are decoded (see _emit_tdm_gather_store_s1 below).

        def silu(x):
            t = x * (-1.4426950408889634)
            emu = rocdl.exp2(T.f32, t)
            den = 1.0 + emu
            sig = rocdl.rcp(T.f32, den)
            return x * sig

        def make_desc_a(k_base):
            return k_base / arith.index(PACK_FACTOR_A)

        # TDM gather for A data
        _use_tdm_gather_a = bool(use_tdm_gather)

        def issue_a_load(k_packed_base, target_lds):
            total = int(tile_m * packed_tile_k_a)
            rounds = (total + block_threads - 1) // block_threads
            for it in range(rounds):
                elem = tx + fx.Index(it * block_threads)
                in_range = arith.cmpi(arith.CmpIPredicate.ult, arith.index_cast(T.i32, elem), arith.constant(total, type=T.i32))
                _if_elem = scf.IfOp(in_range)
                with ir.InsertionPoint(_if_elem.then_block):
                    row = elem // arith.index(int(packed_tile_k_a))
                    col = elem % arith.index(int(packed_tile_k_a))
                    # Use preloaded lds_tid instead of per-thread buffer_load(sorted_rsrc, ...).
                    # Invalid rows were pre-filled with sentinel 0xFFFFFFFF at preload, so
                    # tok=0xFFFFFF will make tok_ok=false for them.
                    fused = _load_fused_from_lds(row)
                    tok = fused & arith.constant((1 << 24) - 1, type=T.i32)
                    tok_ok = arith.cmpi(arith.CmpIPredicate.ult, tok, i32_tokens_in)
                    load_ok = tok_ok
                    x_idx = tok * arith.constant(K_packed_a, type=T.i32) + arith.index_cast(T.i32, k_packed_base + col)
                    x_idx_safe = arith.select(load_ok, x_idx, arith.constant(0, type=T.i32))
                    x_val = arith.select(load_ok, buffer_ops.buffer_load(x_rsrc, x_idx_safe, vec_width=1, dtype=T.i8), arith.constant(0, type=T.i8))
                    lds_idx = row * arith.index(lds_a_stride_bytes) + col
                    v1 = vector.from_elements(T.vec(1, T.i8), [x_val])
                    vector.store(v1, target_lds, [lds_idx], alignment=1)
                    scf.YieldOp([])

        # Pre-compute token row indices for ALL tile_m rows (once, outside K-loop).
        # _a_tok_ids[i] = token_id for TDM gather A load
        # _a_out_row_ids[i] = tok * topk + slot for TDM gather store output
        _a_tok_ids = []
        _a_out_row_ids = []
        _a_load_valids = []
        _a_store_valids = []

        def _sum_i32_values(_vals):
            _acc = arith.constant(0, type=T.i32)
            for _vi in range_constexpr(len(_vals)):
                _acc = _acc + _vals[_vi]
            return _acc

        def _preload_sorted_ids_to_lds():
            """Preload tile_m sorted_token_ids entries into ``lds_tid`` (once per CTA).

            Row ``ri`` (ri in ``[0, tile_m)``) gets the raw i32 from
            ``sorted_token_ids[by * tile_m + ri]`` when that row is both inside
            the block's route slot range and the valid prefix; otherwise the
            sentinel ``0xFFFFFFFF`` is stored so that downstream
            ``tok = fused & 0xFFFFFF`` / ``slot = fused >> 24`` decoding makes
            ``tok_ok`` and ``slot_ok1`` naturally false, eliminating the need
            for separate ``row_in_route`` / ``row_in_valid`` guards at every
            consumer site.
            """
            _tid_in_range = arith.cmpi(
                arith.CmpIPredicate.ult, tx, fx.Index(int(tile_m)))
            _if_tid = scf.IfOp(_tid_in_range)
            with ir.InsertionPoint(_if_tid.then_block):
                _tx_i32 = arith.index_cast(T.i32, tx)
                _sorted_row = by * fx.Index(int(tile_m)) + tx
                _sorted_i32 = arith.index_cast(T.i32, _sorted_row)
                _in_route = arith.cmpi(
                    arith.CmpIPredicate.ult,
                    _tx_i32,
                    arith.constant(int(route_tile_m), type=T.i32),
                )
                _in_valid = arith.cmpi(
                    arith.CmpIPredicate.slt, _sorted_i32, num_valid_i32)
                _row_valid = arith.andi(_in_route, _in_valid)
                _row_safe_i32 = arith.select(
                    _row_valid, _sorted_i32, block_row_start)
                _raw = buffer_ops.buffer_load(
                    sorted_rsrc, _row_safe_i32, vec_width=1, dtype=T.i32)
                _sentinel = arith.constant(-1, type=T.i32)  # 0xFFFFFFFF
                _val = arith.select(_row_valid, _raw, _sentinel)
                _vec1 = vector.from_elements(T.vec(1, T.i32), [_val])
                vector.store(_vec1, lds_tid, [tx], alignment=4)
                scf.YieldOp([])
            workgroup_barrier(use_cluster=use_cluster)

        def _load_fused_from_lds(row_index):
            """Load the cached ``fused`` i32 for a row (``0 <= row_index < tile_m``).

            ``row_index`` may be a Python int (compile-time constant) or an
            index-typed SSA value — both map to a single ``ds_read_b32``.
            Invalid rows were pre-filled with ``0xFFFFFFFF`` at preload time.
            """
            if isinstance(row_index, int):
                row_index = arith.index(row_index)
            return memref.load(lds_tid, [row_index])

        def _precompute_a_row_indices():
            """Decode per-row token/slot meta from ``lds_tid`` into sgpr lists.

            Reads the preloaded i32 for each ``ri`` via a uniform ``ds_read_b32``
            (one per row for the whole wave), then ``readfirstlane`` to produce
            sgpr values used by TDM gather and TDM store. Invalid rows decode
            to ``tok=0xFFFFFF``/``slot=0xFF`` via the sentinel, which the
            ``tok_ok`` / ``slot_ok`` checks below reject.
            """
            _safe_row = arith.constant(0, type=T.i32)
            _one_i32 = arith.constant(1, type=T.i32)
            _zero_i32 = arith.constant(0, type=T.i32)
            for _ri in range_constexpr(int(tile_m)):
                _fused = _load_fused_from_lds(_ri)
                _fused_sgpr = rocdl.readfirstlane(T.i32, _fused)
                _tok = _fused_sgpr & fx.Int32((1 << 24) - 1)
                _slot = _fused_sgpr >> fx.Int32(24)
                _tok_ok = arith.cmpi(arith.CmpIPredicate.ult, _tok, i32_tokens_in)
                _slot_ok0 = arith.cmpi(arith.CmpIPredicate.sge, _slot, fx.Int32(0))
                _slot_ok1 = arith.cmpi(arith.CmpIPredicate.slt, _slot, c_topk_i32)
                _slot_ok = arith.andi(_slot_ok0, _slot_ok1)
                _row_tok_ok = _tok_ok
                _load_valid_i32 = arith.select(_row_tok_ok, _one_i32, _zero_i32)
                _a_load_valids.append(rocdl.readfirstlane(T.i32, _load_valid_i32))
                _tok_safe = arith.select(_row_tok_ok, _tok, _safe_row)
                _tok_sgpr = rocdl.readfirstlane(T.i32, _tok_safe)
                _a_tok_ids.append(_tok_sgpr)
                _out_row = _tok * c_topk_i32 + _slot
                _row_fully_ok = arith.andi(_row_tok_ok, _slot_ok)
                _store_valid_i32 = arith.select(_row_fully_ok, _one_i32, _zero_i32)
                _a_store_valids.append(rocdl.readfirstlane(T.i32, _store_valid_i32))
                _out_row_safe = arith.select(
                    _row_fully_ok, _out_row,
                    _safe_row,
                )
                _out_row_sgpr = rocdl.readfirstlane(T.i32, _out_row_safe)
                _a_out_row_ids.append(_out_row_sgpr)

        _TDM_GATHER_CHUNK = 8
        _TDM_GATHER_GROUPS = (int(tile_m) + _TDM_GATHER_CHUNK - 1) // _TDM_GATHER_CHUNK

        _a_tokens_sgpr = None
        _a_tokens_topk_sgpr = None

        def _get_tokens_sgpr():
            nonlocal _a_tokens_sgpr
            if const_expr(_a_tokens_sgpr is None):
                _tok_i32 = arith.index_cast(T.i32, arith.index_cast(T.index, i32_tokens_in))
                _a_tokens_sgpr = rocdl.readfirstlane(T.i32, _tok_i32)
            return _a_tokens_sgpr

        def _get_tokens_topk_sgpr():
            nonlocal _a_tokens_topk_sgpr
            if const_expr(_a_tokens_topk_sgpr is None):
                _m_i32 = _get_tokens_sgpr() * c_topk_i32
                _a_tokens_topk_sgpr = rocdl.readfirstlane(T.i32, _m_i32)
            return _a_tokens_topk_sgpr

        # Cache of K-invariant pieces of the TDM gather descriptor:
        #   "desc"[_gi][buf_idx] — full TDMGatherDescriptor with addr_lo = base
        #                          (built at global_byte_offset=None),
        #   "pred"[_gi]          — issue predicate (valid_count > 0, wave owner),
        #   "base_addr_lo"[_gi]  — dgroup0.lane2 at global_byte_offset=0,
        # populated once by ``_build_a_gather_base_descs()`` before the K loop
        # so the hot path (``issue_a_load_tdm_gather``) only advances addr_lo
        # via vector.insert each iteration.
        _a_gather_cache = {}

        def _build_a_gather_base_descs(lds_bufs):
            if "desc" in _a_gather_cache:
                return
            _tokens_dim1 = _get_tokens_sgpr()
            _zero_i32 = arith.constant(0, type=T.i32)
            _descs = []
            _preds = []
            _base_addr_lo = []
            for _gi in range_constexpr(_TDM_GATHER_GROUPS):
                _start = _gi * _TDM_GATHER_CHUNK
                _cnt = min(_TDM_GATHER_CHUNK, int(tile_m) - _start)
                _row_indices = _a_tok_ids[_start:_start + _cnt]
                _valid_count = _sum_i32_values(_a_load_valids[_start:_start + _cnt])
                _has_valid = arith.cmpi(arith.CmpIPredicate.sgt, _valid_count, _zero_i32)
                _issue_pred = _has_valid
                if const_expr(wave_specialized_tdm):
                    _gather_owner = _gi % _tdm_loader_waves
                    _is_gather_loader = arith.cmpi(
                        arith.CmpIPredicate.eq,
                        _tdm_wave_id,
                        arith.constant(_gather_owner, type=T.i32),
                    )
                    _issue_pred = arith.andi(_issue_pred, _is_gather_loader)
                _preds.append(_issue_pred)

                _lds_off = fx.Index(_start * lds_a_stride_bytes)
                _per_buf = []
                # NOTE: must use range_constexpr here. The AST rewriter
                # (InsertEmptyYieldForSCFFor) turns a plain `range` inside a
                # kernel body into scf_range -> scf.ForOp, making the loop
                # variable an MLIR induction value (ArithValue) and breaking
                # Python list indexing below.
                for _buf_i in range_constexpr(len(lds_bufs)):
                    _base_desc = tdm_ops.make_tensor_gather_descriptor(
                        global_ptr=arg_x,
                        lds_memref=lds_bufs[_buf_i],
                        row_indices=_row_indices,
                        row_width=int(packed_tile_k_a),
                        tensor_dim0=K_packed_a,
                        tensor_dim1=_tokens_dim1,
                        stride=K_packed_a,
                        elem_bytes=1,
                        pad_interval=int(packed_tile_k_a) if LDS_PAD_A_BYTES > 0 else 0,
                        pad_amount=LDS_PAD_A_BYTES if LDS_PAD_A_BYTES > 0 else 0,
                        index_size=32,
                        gather_tile_dim1=_valid_count,
                        lds_byte_offset=_lds_off,
                        global_byte_offset=None,
                    )
                    _per_buf.append(_base_desc)
                _descs.append(_per_buf)
                # addr_lo is independent of buf_idx (only lds_addr differs), so
                # we can extract it from any buffer's base descriptor.
                _base_addr_lo.append(vector.extract(
                    _per_buf[0].dgroup0,
                    static_position=[2],
                    dynamic_position=[],
                ))

            _a_gather_cache["desc"] = _descs
            _a_gather_cache["pred"] = _preds
            _a_gather_cache["base_addr_lo"] = _base_addr_lo

        def issue_a_load_tdm_gather(k_base, buf_idx):
            """Hot path: advance addr_lo on the precomputed gather descriptor.

            Requires ``_build_a_gather_base_descs(lds_bufs)`` to have been
            called once before the K loop with the matching LDS buffer list.
            """
            k_packed_base = k_base if PACK_FACTOR_A == 1 else k_base // fx.Index(PACK_FACTOR_A)
            _k_byte_off_i32 = arith.index_cast(T.i32, k_packed_base)
            _descs = _a_gather_cache["desc"]
            _preds = _a_gather_cache["pred"]
            _base_addr_lo = _a_gather_cache["base_addr_lo"]
            for _gi in range_constexpr(_TDM_GATHER_GROUPS):
                _if_issue = scf.IfOp(_preds[_gi])
                with ir.InsertionPoint(_if_issue.then_block):
                    _curr_addr_lo = arith.addi(
                        _base_addr_lo[_gi], _k_byte_off_i32)
                    tdm_ops.tensor_load_gather(
                        tdm_ops.update_tensor_gather_descriptor_addr_lo(
                            _descs[_gi][buf_idx],
                            _curr_addr_lo,
                        )
                    )
                    scf.YieldOp([])

        # Cache of K-invariant 2D B / B-scale descriptors used by
        # ``_issue_b_tdm_only``. Each entry stores a base TDMDescriptor2D
        # built at k_base=0 plus its extracted scalar addr_lo, so the hot
        # path only computes ``curr_addr_lo = base_addr_lo + k_off`` and
        # calls ``update_tensor_descriptor_2d_addr_lo`` -- mirroring the
        # hoist that wave_specialized_tdm already does internally via
        # ``_active_stage_desc_base``. ``_build_b_base_descs()`` is closed
        # over later-defined names (``_stage1_pair_row_base``, the
        # ``make_desc_b*`` helpers, and the various ``lds_b*_bufs`` lists);
        # those names are resolved at *call* time inside ``_if_blk``.
        _b_desc_cache = {}

        def _extract_desc_addr_lo(desc):
            return vector.extract(
                desc.dgroup0,
                static_position=[2],
                dynamic_position=[],
            )

        def _build_b_base_descs():
            if "ready" in _b_desc_cache:
                return
            _zero_k = arith.index(0)
            if const_expr(_merge_gate_up_tdm):
                _n_pair = _stage1_pair_row_base()
                _bg_pair = [
                    make_desc_b_pair(lds_bg_pair_bufs[i], _n_pair, _zero_k)
                    for i in range_constexpr(_nb)
                ]
                _bs_pair = [
                    make_desc_bs_pair(lds_bs_pair_bufs[i], _n_pair, _zero_k)
                    for i in range_constexpr(_nb)
                ]
                _b_desc_cache["bg_pair"] = _bg_pair
                _b_desc_cache["bs_pair"] = _bs_pair
                _b_desc_cache["bg_pair_addr_lo"] = [
                    _extract_desc_addr_lo(d) for d in _bg_pair
                ]
                _b_desc_cache["bs_pair_addr_lo"] = [
                    _extract_desc_addr_lo(d) for d in _bs_pair
                ]
            else:
                _eid_row = (
                    arith.index_cast(T.index, eid_i32)
                    * arith.index(int(2 * N))
                )
                _n_gate = _eid_row + blk_n
                _n_up = _eid_row + blk_n + arith.index(int(N))
                _bg = [
                    make_desc_b(lds_bg_bufs[i], _n_gate, _zero_k)
                    for i in range_constexpr(_nb)
                ]
                _bu = [
                    make_desc_b(lds_bu_bufs[i], _n_up, _zero_k)
                    for i in range_constexpr(_nb)
                ]
                _bs = [
                    make_desc_bs(lds_bs_bufs[i], _n_gate, _zero_k)
                    for i in range_constexpr(_nb)
                ]
                _bsu = [
                    make_desc_bs(lds_bsu_bufs[i], _n_up, _zero_k)
                    for i in range_constexpr(_nb)
                ]
                _b_desc_cache["bg"] = _bg
                _b_desc_cache["bu"] = _bu
                _b_desc_cache["bs"] = _bs
                _b_desc_cache["bsu"] = _bsu
                _b_desc_cache["bg_addr_lo"] = [_extract_desc_addr_lo(d) for d in _bg]
                _b_desc_cache["bu_addr_lo"] = [_extract_desc_addr_lo(d) for d in _bu]
                _b_desc_cache["bs_addr_lo"] = [_extract_desc_addr_lo(d) for d in _bs]
                _b_desc_cache["bsu_addr_lo"] = [_extract_desc_addr_lo(d) for d in _bsu]
            _b_desc_cache["ready"] = True

        def _b_data_k_byte_off(k_base):
            # Byte offset along the fastest axis for a B-data descriptor:
            #   non-fp4 / merged pair : (k_base / PACK_FACTOR_B) * 16 bytes
            #   fp4                   : (k_base / PACK_FACTOR_B) bytes
            # Matches `make_desc_b` / `make_desc_b_pair` global_offset math
            # (elem_bytes=1 there, so element offset == byte offset).
            _k_packed_b = (
                k_base if PACK_FACTOR_B == 1
                else k_base // fx.Index(PACK_FACTOR_B)
            )
            if const_expr(is_fp4):
                return arith.index_cast(T.i32, _k_packed_b)
            return arith.index_cast(
                T.i32, _k_packed_b * fx.Index(16))

        def _b_scale_k_byte_off(k_base):
            # B-scale fastest-axis offset: k_base / SCALE_BLOCK bytes.
            return arith.index_cast(
                T.i32, k_base // fx.Index(SCALE_BLOCK))

        def make_desc_as(k_base):
            return k_base / arith.index(SCALE_BLOCK)

        def issue_as_load(k_scale_base, target_lds):
            """Vectorised scalar A-scale loader (Option B).

            Each thread loads one ``SCALES_PER_WMMA``-sized chunk (4 bytes)
            and writes it either to the row-major LDS slot (``is_fp4`` or
            ``wmma_m_rep == 1``) or to the interleaved LDS slot. Avoiding a
            full-row i8 vector load is important for row widths such as 16
            bytes, where LLVM cannot legalize ``v16i8`` raw buffer loads.

            Rare fallback: ``scale_k_per_tile`` not a multiple of 4 falls back
              to the original per-byte loop for correctness.
            """
            _blk_bytes = int(SCALES_PER_WMMA)
            _row_bytes = int(scale_k_per_tile)
            if const_expr(_row_bytes % _blk_bytes == 0 and _row_bytes >= _blk_bytes):
                _blk_vec_type = T.vec(_blk_bytes, T.i8)
                _blks_per_row = _row_bytes // _blk_bytes
                total = int(tile_m) * _blks_per_row
                rounds = (total + block_threads - 1) // block_threads
                for it in range(rounds):
                    elem = tx + fx.Index(it * block_threads)
                    in_range = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        arith.index_cast(T.i32, elem),
                        arith.constant(total, type=T.i32),
                    )
                    _if_elem = scf.IfOp(in_range)
                    with ir.InsertionPoint(_if_elem.then_block):
                        row = elem // arith.index(_blks_per_row)
                        ksc_blk = elem % arith.index(_blks_per_row)
                        fused = _load_fused_from_lds(row)
                        tok = fused & arith.constant((1 << 24) - 1, type=T.i32)
                        tok_ok = arith.cmpi(
                            arith.CmpIPredicate.ult, tok, i32_tokens_in,
                        )
                        if const_expr(_as_layout_rowmajor):
                            lds_idx = (
                                row * arith.index(_row_bytes)
                                + ksc_blk * arith.index(_blk_bytes)
                            )
                        else:
                            warp_row_idx = row / arith.index(warp_tile_m)
                            local_row = row % arith.index(warp_tile_m)
                            lane_row = local_row % arith.index(WMMA_M)
                            local_wm_idx = local_row / arith.index(WMMA_M)
                            global_lds_row = (
                                warp_row_idx * arith.index(WMMA_M) + lane_row
                            )
                            lds_idx = (
                                global_lds_row
                                * arith.index(interleaved_scale_cols_a)
                                + ksc_blk
                                * arith.index(wmma_m_rep * SCALES_PER_WMMA)
                                + local_wm_idx * arith.index(SCALES_PER_WMMA)
                            )
                        _if_ok = scf.IfOp(tok_ok, has_else=True)
                        with ir.InsertionPoint(_if_ok.then_block):
                            chunk_off = (
                                k_scale_base
                                + ksc_blk * arith.index(_blk_bytes)
                            )
                            sx_idx = (
                                tok * arith.constant(K_scale, type=T.i32)
                                + arith.index_cast(T.i32, chunk_off)
                            )
                            sx_raw = buffer_ops.buffer_load(
                                sx_rsrc,
                                arith.shrui(
                                    sx_idx,
                                    arith.constant(2, type=T.i32),
                                ),
                                vec_width=1,
                                dtype=T.i32,
                            )
                            sx_vec = vector.bitcast(
                                _blk_vec_type,
                                vector.from_elements(T.vec(1, T.i32), [sx_raw]),
                            )
                            vector.store(
                                sx_vec, target_lds, [lds_idx],
                                alignment=_blk_bytes,
                            )
                            scf.YieldOp([])
                        with ir.InsertionPoint(_if_ok.else_block):
                            fill_vec = vector.bitcast(
                                _blk_vec_type,
                                vector.from_elements(
                                    T.vec(1, T.i32),
                                    [arith.constant(0x7F7F7F7F, type=T.i32)],
                                ),
                            )
                            vector.store(
                                fill_vec, target_lds, [lds_idx],
                                alignment=_blk_bytes,
                            )
                            scf.YieldOp([])
                        scf.YieldOp([])
            else:
                # Rare fallback: keep per-byte loop for scale widths not aligned
                # to 4 bytes (should not happen for gfx1250 MoE MX configs).
                total = int(tile_m * scale_k_per_tile)
                rounds = (total + block_threads - 1) // block_threads
                for it in range(rounds):
                    elem = tx + fx.Index(it * block_threads)
                    in_range = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        arith.index_cast(T.i32, elem),
                        arith.constant(total, type=T.i32),
                    )
                    _if_elem = scf.IfOp(in_range)
                    with ir.InsertionPoint(_if_elem.then_block):
                        row = elem // arith.index(int(scale_k_per_tile))
                        ksc = elem % arith.index(int(scale_k_per_tile))
                        fused = _load_fused_from_lds(row)
                        tok = fused & arith.constant((1 << 24) - 1, type=T.i32)
                        tok_ok = arith.cmpi(
                            arith.CmpIPredicate.ult, tok, i32_tokens_in,
                        )
                        load_ok = tok_ok
                        ksc_off = k_scale_base + ksc
                        sx_idx = tok * arith.constant(K_scale, type=T.i32) + arith.index_cast(T.i32, ksc_off)
                        sx_idx_safe = arith.select(load_ok, sx_idx, arith.constant(0, type=T.i32))
                        sx_val = arith.select(
                            load_ok,
                            buffer_ops.buffer_load(sx_rsrc, sx_idx_safe, vec_width=1, dtype=T.i8),
                            arith.constant(127, type=T.i8),
                        )
                        if is_fp4:
                            lds_idx = row * arith.index(int(scale_k_per_tile)) + ksc
                        else:
                            warp_row_idx = row / arith.index(warp_tile_m)
                            local_row = row % arith.index(warp_tile_m)
                            lane_row = local_row % arith.index(WMMA_M)
                            local_wm_idx = local_row / arith.index(WMMA_M)
                            global_lds_row = warp_row_idx * arith.index(WMMA_M) + lane_row
                            ksc_blk = ksc / arith.index(SCALES_PER_WMMA)
                            ksc_sub = ksc % arith.index(SCALES_PER_WMMA)
                            lds_idx = (
                                global_lds_row * arith.index(interleaved_scale_cols_a)
                                + ksc_blk * arith.index(wmma_m_rep * SCALES_PER_WMMA)
                                + local_wm_idx * arith.index(SCALES_PER_WMMA)
                                + ksc_sub
                            )
                        v1 = vector.from_elements(T.vec(1, T.i8), [sx_val])
                        vector.store(v1, target_lds, [lds_idx], alignment=1)
                        scf.YieldOp([])

        def issue_as_load_tdm_gather(k_scale_base, target_lds):
            """TDM-gather A-scale loader (Option A).

            Issues one TDM gather per 8-row group (``_TDM_GATHER_GROUPS``
            total), each covering up to 8 rows × ``scale_k_per_tile`` bytes.
            Reuses the ``_a_tok_ids`` SGPR cache built by
            ``_precompute_a_row_indices()`` for the A-data path, so no extra
            scalar loads of ``sorted_rsrc`` are issued here. Completion is
            tracked via ``tdm_cnt`` instead of ``ds_cnt``, eliminating the
            ``s_wait_dscnt 0`` stall cluster previously caused by per-byte
            ``buffer_load`` + ``ds_write_b8`` on the scalar path.

            Pre-conditions (enforced by the gating above):
              - ``use_tdm_gather=True`` (otherwise ``_a_tok_ids`` is empty).
              - Row-major LDS scale layout (``is_fp4`` or ``wmma_m_rep == 1``).
              - ``scale_k_per_tile`` is a positive multiple of 4 (TDM row_width
                hardware alignment).
            """
            _as_row_bytes = int(scale_k_per_tile)
            _tokens_dim1 = _get_tokens_sgpr()
            _zero_i32 = arith.constant(0, type=T.i32)
            for _gi in range_constexpr(_TDM_GATHER_GROUPS):
                _start = _gi * _TDM_GATHER_CHUNK
                _cnt = min(_TDM_GATHER_CHUNK, int(tile_m) - _start)
                _row_indices = _a_tok_ids[_start:_start + _cnt]
                _valid_count = _sum_i32_values(_a_load_valids[_start:_start + _cnt])
                _lds_off = fx.Index(_start * _as_row_bytes)
                _has_valid = arith.cmpi(
                    arith.CmpIPredicate.sgt, _valid_count, _zero_i32,
                )
                _issue_pred = _has_valid
                if wave_specialized_tdm:
                    _gather_owner = _gi % _tdm_loader_waves
                    _is_gather_loader = arith.cmpi(
                        arith.CmpIPredicate.eq,
                        _tdm_wave_id,
                        arith.constant(_gather_owner, type=T.i32),
                    )
                    _issue_pred = arith.andi(_issue_pred, _is_gather_loader)
                _if_issue = scf.IfOp(_issue_pred)
                with ir.InsertionPoint(_if_issue.then_block):
                    desc = tdm_ops.make_tensor_gather_descriptor(
                        global_ptr=arg_scale_x,
                        lds_memref=target_lds,
                        row_indices=_row_indices,
                        row_width=_as_row_bytes,
                        tensor_dim0=int(K_scale),
                        tensor_dim1=_tokens_dim1,
                        stride=int(K_scale),
                        elem_bytes=1,
                        pad_interval=0,
                        pad_amount=0,
                        index_size=32,
                        gather_tile_dim1=_valid_count,
                        lds_byte_offset=_lds_off,
                        global_byte_offset=k_scale_base,
                    )
                    tdm_ops.tensor_load_gather(desc)
                    scf.YieldOp([])

        def make_desc_b(lds_b_mem, n_off, k_base):
            if const_expr(is_fp4):
                return tdm_ops.make_tensor_descriptor_2d(
                    global_ptr=arg_w, lds_memref=lds_b_mem,
                    global_offset=(n_off, k_base / arith.index(PACK_FACTOR_B)),
                    tensor_shape=(int(tile_n), int(packed_tile_k_b)),
                    strides=(K_packed_b, 1),
                    tile_shape=(int(tile_n), int(packed_tile_k_b)),
                    elem_bytes=1, pad_interval=int(packed_tile_k_b), pad_amount=LDS_PAD_B_BYTES,
                    num_warps=tdm_desc_num_warps, workgroup_mask=b_mcast_mask)
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_w, lds_memref=lds_b_mem,
                global_offset=(n_off / arith.index(16), (k_base / arith.index(PACK_FACTOR_B)) * arith.index(16)),
                tensor_shape=(int(experts * (2 * N) // 16), int(K_packed_b * 16)),
                strides=(K_packed_b * 16, 1),
                tile_shape=(int(tile_n // 16), int(packed_tile_k_b * 16)),
                elem_bytes=1,
                pad_interval=0, pad_amount=0,
                num_warps=tdm_desc_num_warps,
                workgroup_mask=b_mcast_mask)

        def make_desc_b_pair(lds_b_mem, n_off, k_base):
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_w, lds_memref=lds_b_mem,
                global_offset=(n_off / arith.index(16), (k_base / arith.index(PACK_FACTOR_B)) * arith.index(16)),
                tensor_shape=(int(experts * (2 * N) // 16), int(K_packed_b * 16)),
                strides=(K_packed_b * 16, 1),
                tile_shape=(int((2 * tile_n) // 16), int(packed_tile_k_b * 16)),
                elem_bytes=1,
                pad_interval=0, pad_amount=0,
                num_warps=tdm_desc_num_warps,
                workgroup_mask=b_mcast_mask)

        def make_desc_bs(lds_bs_mem, n_off, k_base):
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_scale_w, lds_memref=lds_bs_mem,
                global_offset=(n_off, k_base / arith.index(SCALE_BLOCK)),
                tensor_shape=(int(tile_n), int(scale_k_per_tile)),
                strides=(K_scale, 1),
                tile_shape=(int(tile_n), int(scale_k_per_tile)),
                elem_bytes=1, pad_interval=0, pad_amount=0,
                num_warps=tdm_desc_num_warps, workgroup_mask=b_mcast_mask)

        def make_desc_bs_pair(lds_bs_mem, n_off, k_base):
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_scale_w, lds_memref=lds_bs_mem,
                global_offset=(n_off, k_base / arith.index(SCALE_BLOCK)),
                tensor_shape=(int(2 * tile_n), int(scale_k_per_tile)),
                strides=(K_scale, 1),
                tile_shape=(int(2 * tile_n), int(scale_k_per_tile)),
                elem_bytes=1, pad_interval=0, pad_amount=0,
                num_warps=tdm_desc_num_warps, workgroup_mask=b_mcast_mask)

        def _stage1_pair_row_base():
            _eid_row = arith.index_cast(T.index, eid_i32) * arith.index(int(2 * N))
            _tile_idx = blk_n / arith.index(int(tile_n))
            return _eid_row + _tile_idx * arith.index(int(2 * tile_n))

        _ldrs = _make_mxscale_data_loaders(
            tiling=tp, warp_m_base=warp_m_base, warp_n_base=warp_n_base,
            wave_n_idx=wave_n_idx, lane16=lane16, lane_kgrp=lane_kgrp,
            ir=ir, arith=arith, vector=vector, llvm_dialect=llvm_dialect,
            T=T, range_constexpr=range_constexpr,
        )
        _lds_load_b128 = _ldrs["_lds_load_b128"]
        load_data_frag = _ldrs["load_data_frag"]
        load_b_frag = _ldrs["load_b_frag"]
        load_scale_i32 = _ldrs["load_scale_i32"]
        _precompute_a_data_bases = _ldrs["_precompute_a_data_bases"]
        _precompute_b_data_bases = _ldrs["_precompute_b_data_bases"]
        _precompute_a_scale_lane_bases = _ldrs["_precompute_a_scale_lane_bases"]
        _precompute_b_scale_lane_bases = _ldrs["_precompute_b_scale_lane_bases"]
        load_scale_b128 = _ldrs["load_scale_b128"]

        acc_zero = arith.constant_vector(0.0, T.vec(ACC_VEC_SIZE, T.f32))
        acc_g = [acc_zero] * n_accs
        acc_u = [acc_zero] * n_accs

        _if_blk = scf.IfOp(block_ok)
        with ir.InsertionPoint(_if_blk.then_block):
            _preload_sorted_ids_to_lds()
            if const_expr(_use_tdm_gather_a or bool(use_tdm_store)):
                _precompute_a_row_indices()
            a_data_bases = _precompute_a_data_bases()
            b_data_bases = _precompute_b_data_bases()
            if const_expr(_merge_gate_up_tdm):
                b_u_data_bases = [
                    _base + arith.index(lds_b_data_bytes)
                    for _base in b_data_bases
                ]
            else:
                b_u_data_bases = b_data_bases
            as_bases = _precompute_a_scale_lane_bases()
            bs_bases = _precompute_b_scale_lane_bases()
            if const_expr(_merge_gate_up_tdm):
                bsu_bases = [
                    _base + arith.index(lds_b_scale_bytes)
                    for _base in bs_bases
                ]
            else:
                bsu_bases = bs_bases
            _use_scheduled_compute = _use_pipeline and not is_fp4
            _front_wm = (wmma_m_rep + 1) // 2
            _back_wm = wmma_m_rep - _front_wm
            _front_wmma = 2 * _front_wm * wmma_n_rep
            _back_wmma = 2 * _back_wm * wmma_n_rep
            _b_frag_ds_loads_per_wn = 2 if is_a8w4 else 4
            _a_scale_ds_loads = wmma_m_rep if is_fp4 else (wmma_m_rep + 3) // 4
            _b_scale_ds_loads = b_scale_load_rep if is_fp4 else wmma_n_rep
            _gate_up_ds_loads = (
                2 * (wmma_n_rep * _b_frag_ds_loads_per_wn + _b_scale_ds_loads)
                + _a_scale_ds_loads
            )

            # ── compute-tile helper (gate + up) ──────────────────────
            def _load_gate_up_b_and_scales(buf_idx, ks):
                if const_expr(_merge_gate_up_tdm):
                    _gate_b_buf = lds_bg_pair_bufs[buf_idx]
                    _up_b_buf = lds_bg_pair_bufs[buf_idx]
                    _gate_bs_buf = lds_bs_pair_bufs[buf_idx]
                    _up_bs_buf = lds_bs_pair_bufs[buf_idx]
                else:
                    _gate_b_buf = lds_bg_bufs[buf_idx]
                    _up_b_buf = lds_bu_bufs[buf_idx]
                    _gate_bs_buf = lds_bs_bufs[buf_idx]
                    _up_bs_buf = lds_bsu_bufs[buf_idx]

                b_g = [load_b_frag(_gate_b_buf, b_data_bases, wn, ks)
                       for wn in range_constexpr(wmma_n_rep)]
                b_u = [load_b_frag(_up_b_buf, b_u_data_bases, wn, ks)
                       for wn in range_constexpr(wmma_n_rep)]
                if const_expr(is_fp4):
                    as_v = [load_scale_i32(lds_as_bufs[buf_idx], as_bases[wm], ks)
                            for wm in range_constexpr(wmma_m_rep)]
                    bs_gv = [load_scale_i32(_gate_bs_buf, bs_bases[bi], ks)
                             for bi in range_constexpr(b_scale_load_rep)]
                    bs_uv = [load_scale_i32(_up_bs_buf, bsu_bases[bi], ks)
                             for bi in range_constexpr(b_scale_load_rep)]
                else:
                    as_v = load_scale_b128(lds_as_bufs[buf_idx], as_bases[0],
                                           wmma_m_rep, ks)
                    bs_gv = [load_scale_i32(_gate_bs_buf, bs_bases[wn], ks)
                             for wn in range_constexpr(wmma_n_rep)]
                    bs_uv = [load_scale_i32(_up_bs_buf, bsu_bases[wn], ks)
                             for wn in range_constexpr(wmma_n_rep)]
                return b_g, bs_gv, b_u, bs_uv, as_v

            def emit_wmma(accs, wm, wn, a_frag, b_frags, a_scales, b_scales):
                _mxscale_emit_wmma(
                    accs=accs, wm=wm, wn=wn,
                    a_frag=a_frag, b_frags=b_frags,
                    a_scales=a_scales, b_scales=b_scales,
                    is_fp4=is_fp4, is_a8w4=is_a8w4,
                    use_scale_opsel=False,
                    rocdl=rocdl, T=T,
                )

            def _emit_rows(acg_in, acu_in, start_wm, a_frags, b_g, b_u, a_scales, bs_g, bs_u):
                for frag_i in range_constexpr(len(a_frags)):
                    wm = start_wm + frag_i
                    for wn_raw in range_constexpr(wmma_n_rep):
                        wn = (wmma_n_rep - 1 - wn_raw) if (wm % 2 == 1) else wn_raw
                        emit_wmma(acg_in, wm, wn, a_frags[frag_i], b_g, a_scales, bs_g)
                        emit_wmma(acu_in, wm, wn, a_frags[frag_i], b_u, a_scales, bs_u)

            def _compute_k_tile(acg, acu, buf_idx, mid_compute_callback=None):
                _mid_emit_ks = 0
                if const_expr(k_wmma_steps > 1):
                    _mid_emit_wm = wmma_m_rep - 1
                    _mid_emit_wn = wmma_n_rep - 1
                else:
                    _front_wn = (wmma_n_rep + 1) // 2
                    if const_expr(wmma_m_rep > 1):
                        _mid_emit_wm = _front_wm - 1
                        _mid_emit_wn = wmma_n_rep - 1
                    else:
                        _mid_emit_wm = 0
                        _mid_emit_wn = _front_wn - 1
                _did_mid = False
                for ks in range_constexpr(k_wmma_steps):
                    b_g, bs_gv, b_u, bs_uv, as_v = _load_gate_up_b_and_scales(buf_idx, ks)
                    for wm in range_constexpr(wmma_m_rep):
                        a_frag = load_data_frag(lds_ag_bufs[buf_idx],
                                                a_data_bases[wm], ks)
                        for wn_raw in range_constexpr(wmma_n_rep):
                            wn = (wmma_n_rep - 1 - wn_raw) if (wm % 2 == 1) else wn_raw
                            emit_wmma(acg, wm, wn, a_frag, b_g, as_v, bs_gv)
                            emit_wmma(acu, wm, wn, a_frag, b_u, as_v, bs_uv)
                            if const_expr(
                                not _did_mid
                                and mid_compute_callback is not None
                                and ks == _mid_emit_ks
                                and wm == _mid_emit_wm
                                and wn == _mid_emit_wn
                            ):
                                mid_compute_callback()
                                _did_mid = True
                return acg, acu

            def _a_streaming_compute(
                acg,
                acu,
                buf_idx,
                b_g,
                bs_gv,
                b_u,
                bs_uv,
                as_v,
                ks,
                next_bs_info=None,
                mid_compute_callback=None,
            ):
                next_result = None
                a_frags_front = [
                    load_data_frag(lds_ag_bufs[buf_idx], a_data_bases[wm], ks)
                    for wm in range_constexpr(_front_wm)
                ]
                _use_partial_drain = (
                    next_bs_info is not None
                    and _front_wm * wmma_n_rep >= 4
                )

                if const_expr(_use_partial_drain):
                    _next_buf_idx, _next_ks = next_bs_info
                    next_result = _load_gate_up_b_and_scales(_next_buf_idx, _next_ks)
                    rocdl.s_wait_dscnt(_gate_up_ds_loads)
                else:
                    rocdl.s_wait_dscnt(0)

                _emit_rows(acg, acu, 0, a_frags_front, b_g, b_u, as_v, bs_gv, bs_uv)

                if const_expr(mid_compute_callback is not None):
                    rocdl.sched_barrier(0)
                    mid_compute_callback()

                if const_expr(_back_wm > 0):
                    a_frags_back = [
                        load_data_frag(
                            lds_ag_bufs[buf_idx],
                            a_data_bases[_front_wm + h],
                            ks,
                        )
                        for h in range_constexpr(_back_wm)
                    ]
                    _back_drain = _gate_up_ds_loads if _use_partial_drain else 0
                    rocdl.s_wait_dscnt(_back_drain)
                    _emit_rows(
                        acg,
                        acu,
                        _front_wm,
                        a_frags_back,
                        b_g,
                        b_u,
                        as_v,
                        bs_gv,
                        bs_uv,
                    )

                if const_expr(not _use_partial_drain and next_bs_info is not None):
                    _next_buf_idx, _next_ks = next_bs_info
                    next_result = _load_gate_up_b_and_scales(_next_buf_idx, _next_ks)
                return acg, acu, next_result

            def _compute_k_tile_scheduled(acg, acu, buf_idx, mid_compute_callback=None):
                current_g = list(acg)
                current_u = list(acu)
                if const_expr(k_wmma_steps == 1):
                    b_g, bs_gv, b_u, bs_uv, as_v = _load_gate_up_b_and_scales(buf_idx, 0)
                    current_g, current_u, _ = _a_streaming_compute(
                        current_g, current_u, buf_idx,
                        b_g, bs_gv, b_u, bs_uv, as_v, 0,
                        mid_compute_callback=mid_compute_callback,
                    )
                else:
                    b_g, bs_gv, b_u, bs_uv, as_v = _load_gate_up_b_and_scales(buf_idx, 0)
                    for ks in range_constexpr(k_wmma_steps - 1):
                        _mid_cb = mid_compute_callback if ks == 0 else None
                        current_g, current_u, _next = _a_streaming_compute(
                            current_g, current_u, buf_idx,
                            b_g, bs_gv, b_u, bs_uv, as_v, ks,
                            next_bs_info=(buf_idx, ks + 1),
                            mid_compute_callback=_mid_cb,
                        )
                        b_g, bs_gv, b_u, bs_uv, as_v = _next
                    current_g, current_u, _ = _a_streaming_compute(
                        current_g, current_u, buf_idx,
                        b_g, bs_gv, b_u, bs_uv, as_v,
                        k_wmma_steps - 1,
                    )
                return current_g, current_u

            def _hot_loop_scheduler_scheduled():
                if const_expr(not _use_scheduled_compute):
                    return
                _front_a_loads = _front_wm * DS_LOADS_PER_A_FRAG
                _back_a_loads = _back_wm * DS_LOADS_PER_A_FRAG
                for _ks in range_constexpr(k_wmma_steps):
                    if const_expr(_ks == 0):
                        rocdl.sched_dsrd(_gate_up_ds_loads + _front_a_loads)
                    else:
                        rocdl.sched_dsrd(_front_a_loads)
                    rocdl.sched_mfma(_front_wmma)
                    if const_expr(_back_wmma > 0):
                        rocdl.sched_dsrd(_back_a_loads)
                        rocdl.sched_mfma(_back_wmma)
                    if const_expr(_ks < k_wmma_steps - 1):
                        rocdl.sched_dsrd(_gate_up_ds_loads)
                rocdl.sched_barrier(0)

            if const_expr(wave_specialized_tdm):
                _tdm_wave_id = rocdl.wave_id()
                _loader_waves = _tdm_loader_waves
                _is_loader_wave = arith.cmpi(
                    arith.CmpIPredicate.ult,
                    _tdm_wave_id,
                    arith.constant(_loader_waves, type=T.i32),
                )
                _tdm_pred = arith.constant(1, type=T.i32)

                def _select_wave_tdm_value(*values):
                    if const_expr(len(values) != _loader_waves):
                        raise ValueError(
                            f"expected {_loader_waves} wave-specialized TDM values, got {len(values)}"
                        )
                    _selected = values[-1]
                    for _sel_idx in range_constexpr(_loader_waves - 1):
                        _value_idx = _loader_waves - 2 - _sel_idx
                        _is_wave = arith.cmpi(
                            arith.CmpIPredicate.eq,
                            _tdm_wave_id,
                            arith.constant(_value_idx, type=T.i32),
                        )
                        _selected = arith.select(_is_wave, values[_value_idx], _selected)
                    return _selected

                def _tdm_desc_lds_addr(desc):
                    return vector.extract(
                        desc.dgroup0,
                        static_position=[1],
                        dynamic_position=[],
                    )

                def _tdm_desc_addr_lo(desc):
                    return vector.extract(
                        desc.dgroup0,
                        static_position=[2],
                        dynamic_position=[],
                    )

                def _tdm_desc_addr_hi(desc):
                    return vector.extract(
                        desc.dgroup0,
                        static_position=[3],
                        dynamic_position=[],
                    )

                _zero_k_base = arith.index(0)
                _scale_adv_i32 = arith.constant(scale_k_per_tile, type=T.i32)
                if const_expr(_merge_gate_up_tdm):
                    _n_pair_init = _stage1_pair_row_base()
                    _data_adv_i32 = arith.constant(packed_tile_k_b * 16, type=T.i32)

                    _stages_b_lds_addr = [
                        _tdm_desc_lds_addr(
                            make_desc_b_pair(
                                lds_bg_pair_bufs[i],
                                _n_pair_init,
                                _zero_k_base,
                            )
                        )
                        for i in range_constexpr(_nb)
                    ]
                    _stages_bs_lds_addr = [
                        _tdm_desc_lds_addr(
                            make_desc_bs_pair(
                                lds_bs_pair_bufs[i],
                                _n_pair_init,
                                _zero_k_base,
                            )
                        )
                        for i in range_constexpr(_nb)
                    ]

                    _desc_b_init = make_desc_b_pair(
                        lds_bg_pair_bufs[0],
                        _n_pair_init,
                        _zero_k_base,
                    )
                    _desc_bs_init = make_desc_bs_pair(
                        lds_bs_pair_bufs[0],
                        _n_pair_init,
                        _zero_k_base,
                    )

                    _active_stage_lds_addr = [
                        _select_wave_tdm_value(
                            _stages_b_lds_addr[i],
                            _stages_bs_lds_addr[i],
                        )
                        for i in range_constexpr(_nb)
                    ]
                    _active_addr_lo = _select_wave_tdm_value(
                        _tdm_desc_addr_lo(_desc_b_init),
                        _tdm_desc_addr_lo(_desc_bs_init),
                    )
                    _active_addr_hi = _select_wave_tdm_value(
                        _tdm_desc_addr_hi(_desc_b_init),
                        _tdm_desc_addr_hi(_desc_bs_init),
                    )
                    _active_dgroup1 = _select_wave_tdm_value(
                        _desc_b_init.dgroup1,
                        _desc_bs_init.dgroup1,
                    )
                    _active_adv_i32 = _select_wave_tdm_value(
                        _data_adv_i32,
                        _scale_adv_i32,
                    )
                else:
                    _eid_row = (
                        arith.index_cast(T.index, eid_i32)
                        * arith.index(int(2 * N))
                    )
                    _n_gate_init = _eid_row + blk_n
                    _n_up_init = _eid_row + blk_n + arith.index(int(N))
                    _data_adv_i32 = arith.constant(
                        packed_tile_k_b if is_fp4 else packed_tile_k_b * 16,
                        type=T.i32,
                    )

                    _stages_bg_lds_addr = [
                        _tdm_desc_lds_addr(
                            make_desc_b(
                                lds_bg_bufs[i],
                                _n_gate_init,
                                _zero_k_base,
                            )
                        )
                        for i in range_constexpr(_nb)
                    ]
                    _stages_bu_lds_addr = [
                        _tdm_desc_lds_addr(
                            make_desc_b(
                                lds_bu_bufs[i],
                                _n_up_init,
                                _zero_k_base,
                            )
                        )
                        for i in range_constexpr(_nb)
                    ]
                    _stages_bs_lds_addr = [
                        _tdm_desc_lds_addr(
                            make_desc_bs(
                                lds_bs_bufs[i],
                                _n_gate_init,
                                _zero_k_base,
                            )
                        )
                        for i in range_constexpr(_nb)
                    ]
                    _stages_bsu_lds_addr = [
                        _tdm_desc_lds_addr(
                            make_desc_bs(
                                lds_bsu_bufs[i],
                                _n_up_init,
                                _zero_k_base,
                            )
                        )
                        for i in range_constexpr(_nb)
                    ]

                    _desc_bg_init = make_desc_b(
                        lds_bg_bufs[0],
                        _n_gate_init,
                        _zero_k_base,
                    )
                    _desc_bu_init = make_desc_b(
                        lds_bu_bufs[0],
                        _n_up_init,
                        _zero_k_base,
                    )
                    _desc_bs_init = make_desc_bs(
                        lds_bs_bufs[0],
                        _n_gate_init,
                        _zero_k_base,
                    )
                    _desc_bsu_init = make_desc_bs(
                        lds_bsu_bufs[0],
                        _n_up_init,
                        _zero_k_base,
                    )

                    _active_stage_lds_addr = [
                        _select_wave_tdm_value(
                            _stages_bg_lds_addr[i],
                            _stages_bu_lds_addr[i],
                            _stages_bs_lds_addr[i],
                            _stages_bsu_lds_addr[i],
                        )
                        for i in range_constexpr(_nb)
                    ]
                    _active_addr_lo = _select_wave_tdm_value(
                        _tdm_desc_addr_lo(_desc_bg_init),
                        _tdm_desc_addr_lo(_desc_bu_init),
                        _tdm_desc_addr_lo(_desc_bs_init),
                        _tdm_desc_addr_lo(_desc_bsu_init),
                    )
                    _active_addr_hi = _select_wave_tdm_value(
                        _tdm_desc_addr_hi(_desc_bg_init),
                        _tdm_desc_addr_hi(_desc_bu_init),
                        _tdm_desc_addr_hi(_desc_bs_init),
                        _tdm_desc_addr_hi(_desc_bsu_init),
                    )
                    _active_dgroup1 = _select_wave_tdm_value(
                        _desc_bg_init.dgroup1,
                        _desc_bu_init.dgroup1,
                        _desc_bs_init.dgroup1,
                        _desc_bsu_init.dgroup1,
                    )
                    _active_adv_i32 = _select_wave_tdm_value(
                        _data_adv_i32,
                        _data_adv_i32,
                        _scale_adv_i32,
                        _scale_adv_i32,
                    )

                # Pre-build per-stage TDMDescriptor2D bases at addr_lo=0 so the
                # hot path only patches lane 2 via update_tensor_descriptor_2d_addr_lo,
                # avoiding a fresh vector.from_elements + TDMDescriptor2D each issue.
                _tdm_zero_addr_lo = arith.constant(0, type=T.i32)
                _active_stage_desc_base = [
                    tdm_ops.TDMDescriptor2D(
                        vector.from_elements(T.vec(4, T.i32), [
                            _tdm_pred,
                            _active_stage_lds_addr[i],
                            _tdm_zero_addr_lo,
                            _active_addr_hi,
                        ]),
                        _active_dgroup1,
                    )
                    for i in range_constexpr(_nb)
                ]

                def _issue_active_b_tdm_only(stage_idx, curr_addr_lo):
                    _if_loader = scf.IfOp(_is_loader_wave)
                    with ir.InsertionPoint(_if_loader.then_block):
                        tdm_ops.tensor_load_2d(
                            tdm_ops.update_tensor_descriptor_2d_addr_lo(
                                _active_stage_desc_base[stage_idx],
                                curr_addr_lo,
                            )
                        )
                        scf.YieldOp([])
                    _next_addr_lo = arith.addi(curr_addr_lo, _active_adv_i32)
                    return arith.select(
                        _is_loader_wave,
                        _next_addr_lo,
                        curr_addr_lo,
                    )

            if const_expr(_use_tdm_gather_a):
                _build_a_gather_base_descs(lds_ag_bufs)
            # Hoist K-invariant parts of B / B-scale 2D descriptors so the
            # hot K loop only has to advance addr_lo per tile. In
            # wave-specialized mode the hot path goes through
            # ``_issue_active_b_tdm_only`` (which is already hoisted via
            # ``_active_stage_desc_base``) and ``_issue_b_tdm_only`` is only
            # reachable from the tail/non-pipelined paths; skip the build
            # there to avoid emitting dead IR.
            if const_expr(not wave_specialized_tdm):
                _build_b_base_descs()

            # ── pipeline load helpers ─────────────────────────────────
            def _issue_b_tdm_only(k_base, buf_idx):
                _k_data_off = _b_data_k_byte_off(k_base)
                _k_scale_off = _b_scale_k_byte_off(k_base)
                if const_expr(_merge_gate_up_tdm):
                    _bg_addr = arith.addi(
                        _b_desc_cache["bg_pair_addr_lo"][buf_idx], _k_data_off)
                    _bs_addr = arith.addi(
                        _b_desc_cache["bs_pair_addr_lo"][buf_idx], _k_scale_off)
                    tdm_ops.tensor_load_2d(
                        tdm_ops.update_tensor_descriptor_2d_addr_lo(
                            _b_desc_cache["bg_pair"][buf_idx], _bg_addr))
                    tdm_ops.tensor_load_2d(
                        tdm_ops.update_tensor_descriptor_2d_addr_lo(
                            _b_desc_cache["bs_pair"][buf_idx], _bs_addr))
                else:
                    _bg_addr = arith.addi(
                        _b_desc_cache["bg_addr_lo"][buf_idx], _k_data_off)
                    _bu_addr = arith.addi(
                        _b_desc_cache["bu_addr_lo"][buf_idx], _k_data_off)
                    _bs_addr = arith.addi(
                        _b_desc_cache["bs_addr_lo"][buf_idx], _k_scale_off)
                    _bsu_addr = arith.addi(
                        _b_desc_cache["bsu_addr_lo"][buf_idx], _k_scale_off)
                    tdm_ops.tensor_load_2d(
                        tdm_ops.update_tensor_descriptor_2d_addr_lo(
                            _b_desc_cache["bg"][buf_idx], _bg_addr))
                    tdm_ops.tensor_load_2d(
                        tdm_ops.update_tensor_descriptor_2d_addr_lo(
                            _b_desc_cache["bu"][buf_idx], _bu_addr))
                    tdm_ops.tensor_load_2d(
                        tdm_ops.update_tensor_descriptor_2d_addr_lo(
                            _b_desc_cache["bs"][buf_idx], _bs_addr))
                    tdm_ops.tensor_load_2d(
                        tdm_ops.update_tensor_descriptor_2d_addr_lo(
                            _b_desc_cache["bsu"][buf_idx], _bsu_addr))

            def _issue_scalar_loads(k_base, buf_idx):
                if const_expr(_use_tdm_gather_a):
                    issue_a_load_tdm_gather(k_base, buf_idx)
                else:
                    issue_a_load(make_desc_a(k_base), lds_ag_bufs[buf_idx])
                if _use_tdm_gather_as:
                    issue_as_load_tdm_gather(make_desc_as(k_base), lds_as_bufs[buf_idx])
                else:
                    issue_as_load(make_desc_as(k_base), lds_as_bufs[buf_idx])

            def _issue_all_loads(k_base, buf_idx):
                _issue_b_tdm_only(k_base, buf_idx)
                _issue_scalar_loads(k_base, buf_idx)

            def _compute_with_mid_loads(acg, acu, buf_idx, mid_load_callback=None):
                if const_expr(_use_scheduled_compute):
                    return _compute_k_tile_scheduled(
                        acg, acu, buf_idx,
                        mid_compute_callback=mid_load_callback,
                    )
                return _compute_k_tile(
                    acg, acu, buf_idx,
                    mid_compute_callback=mid_load_callback,
                )

            # Helper: apply split-K K-base offset. For non-splitk the
            # compile-time constant expression is returned unchanged so
            # the non-splitk code path is identical.
            def _k_off(static_offset_val):
                if _is_splitk:
                    return k_base_idx + static_offset_val
                return static_offset_val

            # ── main K-dimension reduction ────────────────────────────
            if const_expr(not _use_pipeline):
                if const_expr(wave_specialized_tdm):
                    active_b_addr_lo = _active_addr_lo
                    for kt in range_constexpr(num_k_tiles_per_bz):
                        k_base = _k_off(fx.Index(kt * int(tile_k)))
                        active_b_addr_lo = _issue_active_b_tdm_only(
                            0, active_b_addr_lo)
                        _issue_scalar_loads(k_base, 0)
                        tdm_ops.tensor_wait(0)
                        workgroup_barrier(use_cluster=use_cluster)
                        acc_g, acc_u = _compute_k_tile(acc_g, acc_u, 0)
                        workgroup_barrier(use_cluster=use_cluster)
                else:
                    for kt in range_constexpr(num_k_tiles_per_bz):
                        k_base = _k_off(fx.Index(kt * int(tile_k)))
                        _issue_all_loads(k_base, 0)
                        tdm_ops.tensor_wait(0)
                        workgroup_barrier(use_cluster=use_cluster)
                        acc_g, acc_u = _compute_k_tile(acc_g, acc_u, 0)
                        workgroup_barrier(use_cluster=use_cluster)
            else:
                # ── prologue ──
                if const_expr(wave_specialized_tdm):
                    active_b_addr_lo = _active_addr_lo
                    for _pi in range_constexpr(pre_loaded):
                        active_b_addr_lo = _issue_active_b_tdm_only(
                            _pi, active_b_addr_lo)
                        _issue_scalar_loads(
                            _k_off(fx.Index(_pi * int(tile_k))), _pi)
                else:
                    for _pi in range_constexpr(pre_loaded):
                        _issue_all_loads(
                            _k_off(fx.Index(_pi * int(tile_k))), _pi)
                pipeline_fence(outstanding=0, use_cluster=use_cluster)

                # ── main pipelined loop ──
                if const_expr(loop_iters > 0):
                    if const_expr(wave_specialized_tdm):
                        _init = list(acc_g) + list(acc_u) + [active_b_addr_lo]
                        for _li, _st in fx.range(0, loop_iters, 1, init=_init):
                            _ag = list(_st[:n_accs])
                            _au = list(_st[n_accs:2 * n_accs])
                            _cur_b_addr_lo = _st[2 * n_accs]
                            for _bi in range_constexpr(_nb):
                                _lb = (_bi + _nb - 1) % _nb
                                _kt = (_li * fx.Index(_nb)
                                       + fx.Index(pre_loaded + _bi))
                                _kb = _k_off(_kt * fx.Index(int(tile_k)))
                                pipeline_fence_signal(
                                    outstanding=_fence_outstanding,
                                    use_cluster=use_cluster)
                                pipeline_fence_wait(use_cluster=use_cluster)
                                _cur_b_addr_lo = _issue_active_b_tdm_only(
                                    _lb, _cur_b_addr_lo)

                                def _mid_issue_scalar(_mid_kb=_kb, _mid_lb=_lb):
                                    _issue_scalar_loads(_mid_kb, _mid_lb)

                                if const_expr(_use_scheduled_compute):
                                    rocdl.sched_barrier(0)
                                _ag, _au = _compute_with_mid_loads(
                                    _ag,
                                    _au,
                                    _bi,
                                    _mid_issue_scalar,
                                )
                                if const_expr(_use_scheduled_compute):
                                    _hot_loop_scheduler_scheduled()
                            _res = yield list(_ag) + list(_au) + [_cur_b_addr_lo]
                        acc_g = list(_res[:n_accs])
                        acc_u = list(_res[n_accs:2 * n_accs])
                        active_b_addr_lo = _res[2 * n_accs]
                    else:
                        _init = list(acc_g) + list(acc_u)
                        for _li, _st in fx.range(0, loop_iters, 1, init=_init):
                            _ag = list(_st[:n_accs])
                            _au = list(_st[n_accs:2 * n_accs])
                            for _bi in range_constexpr(_nb):
                                _lb = (_bi + _nb - 1) % _nb
                                _kt = (_li * fx.Index(_nb)
                                       + fx.Index(pre_loaded + _bi))
                                _kb = _k_off(_kt * fx.Index(int(tile_k)))
                                pipeline_fence_signal(
                                    outstanding=_fence_outstanding,
                                    use_cluster=use_cluster)
                                pipeline_fence_wait(use_cluster=use_cluster)
                                _issue_b_tdm_only(_kb, _lb)

                                def _mid_issue_scalar(_mid_kb=_kb, _mid_lb=_lb):
                                    _issue_scalar_loads(_mid_kb, _mid_lb)

                                if const_expr(_use_scheduled_compute):
                                    rocdl.sched_barrier(0)
                                _ag, _au = _compute_with_mid_loads(
                                    _ag,
                                    _au,
                                    _bi,
                                    _mid_issue_scalar,
                                )
                                if const_expr(_use_scheduled_compute):
                                    _hot_loop_scheduler_scheduled()
                            _res = yield list(_ag) + list(_au)
                        acc_g = list(_res[:n_accs])
                        acc_u = list(_res[n_accs:2 * n_accs])

                # ── post-loop fence ──
                if const_expr(loop_iters > 0):
                    pipeline_fence(outstanding=0, use_cluster=use_cluster)
                elif const_expr(use_cluster):
                    gpu.cluster_barrier()

                # ── tail ──
                _tail_li = 0
                _tail_had_load = False
                for _ls, _cs, _out in _tail_plan:
                    if const_expr(_out == -1):
                        if const_expr(_tail_had_load):
                            pipeline_fence(outstanding=0,
                                           use_cluster=use_cluster)
                        if const_expr(_use_scheduled_compute):
                            rocdl.sched_barrier(0)
                            acc_g, acc_u = _compute_k_tile_scheduled(
                                acc_g, acc_u, _cs)
                            _hot_loop_scheduler_scheduled()
                        else:
                            acc_g, acc_u = _compute_k_tile(
                                acc_g, acc_u, _cs)
                    else:
                        pipeline_fence_signal(outstanding=_out,
                                              use_cluster=use_cluster)
                        pipeline_fence_wait(use_cluster=use_cluster)
                        if const_expr(_ls is not None):
                            _tail_had_load = True
                            _tkb = _k_off(fx.Index(
                                (_tail_start + pre_loaded + _tail_li)
                                * int(tile_k)))
                            _tail_li += 1
                            if const_expr(wave_specialized_tdm):
                                active_b_addr_lo = _issue_active_b_tdm_only(
                                    _ls, active_b_addr_lo)
                            else:
                                _issue_b_tdm_only(_tkb, _ls)

                            def _tail_mid_issue_scalar(_mid_kb=_tkb, _mid_ls=_ls):
                                _issue_scalar_loads(_mid_kb, _mid_ls)

                            if const_expr(_use_scheduled_compute):
                                rocdl.sched_barrier(0)
                            acc_g, acc_u = _compute_with_mid_loads(
                                acc_g,
                                acc_u,
                                _cs,
                                _tail_mid_issue_scalar,
                            )
                            if const_expr(_use_scheduled_compute):
                                _hot_loop_scheduler_scheduled()
                        else:
                            if const_expr(_use_scheduled_compute):
                                rocdl.sched_barrier(0)
                                acc_g, acc_u = _compute_k_tile_scheduled(
                                    acc_g, acc_u, _cs)
                                _hot_loop_scheduler_scheduled()
                            else:
                                acc_g, acc_u = _compute_k_tile(
                                    acc_g, acc_u, _cs)

            out_elem_ty = _moe_out_elem_ty(out_dtype, T)

            if const_expr(bool(use_tdm_store)):
                # ── TDM store epilogue: silu(gate)*up → LDS → global (contiguous sorted output) ──
                _scale_per_wm_s1 = []
                for _wm in range_constexpr(wmma_m_rep):
                    _m_off_val = _wm * WMMA_M
                    _row_local = warp_m_base + arith.index(_m_off_val) + lane16
                    _sorted_row = by * arith.index(int(tile_m)) + _row_local
                    _sorted_i32 = arith.index_cast(T.i32, _sorted_row)
                    _row_in_route = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        arith.index_cast(T.i32, _row_local),
                        arith.constant(int(route_tile_m), type=T.i32))
                    if const_expr(bool(doweight_stage1)):
                        _sorted_safe = arith.select(
                            _row_in_route, _sorted_i32,
                            arith.index_cast(T.i32,
                                by * arith.index(int(route_tile_m))))
                        _tw = buffer_ops.buffer_load(
                            tw_rsrc, _sorted_safe, vec_width=1, dtype=T.f32)
                        _sc = arith.select(
                            _row_in_route, _tw,
                            arith.constant(0.0, type=T.f32))
                    else:
                        _sc = arith.select(
                            _row_in_route,
                            arith.constant(1.0, type=T.f32),
                            arith.constant(0.0, type=T.f32))
                    _scale_per_wm_s1.append(_sc)

                if const_expr(d_need_epilogue_fence_s1):
                    pipeline_fence(outstanding=0, use_cluster=use_cluster)
                rocdl.sched_barrier(0)

                for _acc_idx, _vec_base, _m_off, _wn in _sub_tiles:
                    _wm_idx = _m_off // WMMA_M
                    _sc = _scale_per_wm_s1[_wm_idx]
                    _sub8g = _extract_sub8(
                        acc_g[_acc_idx], _vec_base,
                        vector=vector,
                        range_constexpr=range_constexpr,
                        ACC_VEC_SIZE=ACC_VEC_SIZE)
                    _sub8u = _extract_sub8(
                        acc_u[_acc_idx], _vec_base,
                        vector=vector,
                        range_constexpr=range_constexpr,
                        ACC_VEC_SIZE=ACC_VEC_SIZE)
                    _fused = []
                    for _vi in range_constexpr(8):
                        _vg = vector.extract(
                            _sub8g,
                            static_position=[_vi],
                            dynamic_position=[])
                        _vu = vector.extract(
                            _sub8u,
                            static_position=[_vi],
                            dynamic_position=[])
                        _y = silu(_vg) * _vu * _sc
                        _fused.append(_y)
                    _fused_sub8 = vector.from_elements(
                        T.vec(8, T.f32), _fused)
                    _imm = (_m_off * _lds_d_stride_elems_s1
                            + _wn * _n_col_d_elems_s1)
                    store_acc_vec8_to_lds(
                        d_lds_buffer_s1, d_lane_base_s1, _imm,
                        _fused_sub8, out_elem=out_elem_ty)

                rocdl.s_wait_dscnt(0)
                # TDM gather store: each warp stores its warp_tile_m rows
                # to scattered output positions tok*topk+slot.
                _warp_row_start = arith.index_cast(T.i32, warp_m_base)
                _warp_row_start_py = rocdl.readfirstlane(T.i32, _warp_row_start)
                _d_store_chunk = 8  # 32-bit gather mode
                _d_store_groups = (warp_tile_m + _d_store_chunk - 1) // _d_store_chunk
                _tokens_topk_dim1 = _get_tokens_topk_sgpr()
                for _dsi in range_constexpr(_d_store_groups):
                    _ds_start = _dsi * _d_store_chunk
                    _ds_cnt = min(_d_store_chunk, warp_tile_m - _ds_start)
                    # Global output row indices for this group
                    _ds_start_in_tile = _dsi * _d_store_chunk + rocdl.readfirstlane(
                        T.i32, arith.index_cast(T.i32, warp_m_base))
                    # Can't do runtime add on SGPR easily; use compile-time
                    # warp offset from wave_id. But warp_m_base is runtime.
                    # Instead, index _a_out_row_ids which is tile-global.
                    # warp_m_base = wave_m_idx * warp_tile_m (runtime index)
                    # We need _a_out_row_ids[warp_m_base + _ds_start + i]
                    # Since warp_m_base depends on wave_id, we use scf.if
                    # per warp to select the correct slice.
                    # Simpler: for num_warps_m = m_warp, unroll per warp:
                    _ds_indices = []
                    _ds_valids = []
                    for _wi in range_constexpr(int(m_warp)):
                        _tile_row = _wi * warp_tile_m + _ds_start
                        _warp_indices = _a_out_row_ids[_tile_row:_tile_row + _ds_cnt]
                        _warp_valids = _a_store_valids[_tile_row:_tile_row + _ds_cnt]
                        if const_expr(_wi == 0):
                            _ds_indices = list(_warp_indices)
                            _ds_valids = list(_warp_valids)
                        else:
                            _is_this_warp = arith.cmpi(
                                arith.CmpIPredicate.eq,
                                rocdl.wave_id() % fx.Int32(int(n_warp * m_warp) // int(n_warp)),
                                fx.Int32(_wi))
                            # Actually wave_m_idx is the M warp index
                            _is_this_warp = arith.cmpi(
                                arith.CmpIPredicate.eq,
                                arith.index_cast(T.i32, wave_m_idx),
                                fx.Int32(_wi))
                            for _ii in range_constexpr(len(_ds_indices)):
                                _ds_indices[_ii] = arith.select(
                                    _is_this_warp,
                                    _warp_indices[_ii],
                                    _ds_indices[_ii])
                                _ds_valids[_ii] = arith.select(
                                    _is_this_warp,
                                    _warp_valids[_ii],
                                    _ds_valids[_ii])
                    # LDS offset within D buffer for this group
                    _ds_lds_off = arith.index(
                        _ds_start * lds_d_row_stride_s1) + d_warp_off_sgpr_s1
                    # Column offset in output
                    _col_byte_off = (blk_n + warp_n_off_sgpr_s1) * arith.index(elem_bytes_d_s1)
                    # For store direction: TDM ignores pad_enable, so we
                    # expand tile_dim0 to include padding so LDS read
                    # addresses align. tensor_dim0 stays at warp_tile_n so
                    # the extra pad elements hit OOB and are dropped.
                    _pad_elems = LDS_PAD_D_BYTES_s1 // elem_bytes_d_s1
                    _store_tile_w = warp_tile_n + _pad_elems
                    _ds_valid_count = _sum_i32_values(_ds_valids)
                    _zero_i32 = arith.constant(0, type=T.i32)
                    _has_store = arith.cmpi(arith.CmpIPredicate.sgt, _ds_valid_count, _zero_i32)
                    _if_store = scf.IfOp(_has_store)
                    with ir.InsertionPoint(_if_store.then_block):
                        _d_store_desc = tdm_ops.make_tensor_gather_descriptor(
                            global_ptr=arg_out,
                            lds_memref=base_ptr,
                            row_indices=_ds_indices,
                            row_width=_store_tile_w,
                            tensor_dim0=warp_tile_n,
                            tensor_dim1=_tokens_topk_dim1,
                            stride=N,
                            elem_bytes=elem_bytes_d_s1,
                            pad_interval=0,
                            pad_amount=0,
                            index_size=32,
                            gather_tile_dim1=_ds_valid_count,
                            lds_byte_offset=_ds_lds_off,
                            global_byte_offset=_col_byte_off,
                        )
                        tdm_ops.tensor_store_gather(_d_store_desc)
                        scf.YieldOp([])
                tdm_ops.tensor_wait(0)
            else:
                def _load_gate_up_sub8(acc_idx, vec_base):
                    return (
                        _extract_sub8(
                            acc_g[acc_idx], vec_base, vector=vector, range_constexpr=range_constexpr, ACC_VEC_SIZE=ACC_VEC_SIZE
                        ),
                        _extract_sub8(
                            acc_u[acc_idx], vec_base, vector=vector, range_constexpr=range_constexpr, ACC_VEC_SIZE=ACC_VEC_SIZE
                        ),
                    )

                if _is_splitk:
                    # Split-K: atomic-fadd gate/up partials into a
                    # [tokens*topk, 2*inter_dim] buffer. silu/mul and the
                    # routing weight fold in via the external reduction.
                    _emit_stage1_gate_up_splitk_epilogue(
                        sub_tiles=_sub_tiles,
                        by=by,
                        tile_m=int(tile_m),
                        route_tile_m=int(route_tile_m),
                        warp_m_base=warp_m_base,
                        warp_n_base=warp_n_base,
                        blk_n=blk_n,
                        lane16=lane16,
                        lane_kgrp=lane_kgrp,
                        WMMA_N=WMMA_N,
                        i32_tokens_in=i32_tokens_in,
                        i32_inter_in=i32_inter_in,
                        topk=int(topk),
                        num_valid_i32=num_valid_i32,
                        block_row_start=block_row_start,
                        lds_tid=lds_tid,
                        memref=memref,
                        sorted_rsrc=sorted_rsrc,
                        out_rsrc=out_rsrc,
                        out_elem_ty=out_elem_ty,
                        load_gate_up_sub8=_load_gate_up_sub8,
                        ir=ir,
                        fx=fx,
                        arith=arith,
                        buffer_ops=buffer_ops,
                        scf=scf,
                        vector=vector,
                        range_constexpr=range_constexpr,
                        rocdl=rocdl,
                        T=T,
                    )
                else:
                    _emit_stage1_gate_up_epilogue(
                        sub_tiles=_sub_tiles,
                        by=by,
                        tile_m=int(tile_m),
                        route_tile_m=int(route_tile_m),
                        warp_m_base=warp_m_base,
                        warp_n_base=warp_n_base,
                        blk_n=blk_n,
                        lane16=lane16,
                        lane_kgrp=lane_kgrp,
                        WMMA_N=WMMA_N,
                        i32_tokens_in=i32_tokens_in,
                        i32_inter_in=i32_inter_in,
                        topk=int(topk),
                        num_valid_i32=num_valid_i32,
                        block_row_start=block_row_start,
                        lds_tid=lds_tid,
                        memref=memref,
                        sorted_rsrc=sorted_rsrc,
                        tw_rsrc=tw_rsrc,
                        out_rsrc=out_rsrc,
                        doweight_stage1=bool(doweight_stage1),
                        out_elem_ty=out_elem_ty,
                        load_gate_up_sub8=_load_gate_up_sub8,
                        silu_fn=silu,
                        ir=ir,
                        fx=fx,
                        arith=arith,
                        buffer_ops=buffer_ops,
                        scf=scf,
                        vector=vector,
                        range_constexpr=range_constexpr,
                        T=T,
                    )
            scf.YieldOp([])

    @flyc.jit
    def launch_mxscale_stage1_single(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_num_valid_ids: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_inter_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
        stream: fx.Stream,
    ):
        _ = i32_k_in
        ctx = CompilationContext.get_current()
        inter_in = arith.index_cast(T.index, i32_inter_in)
        size_expert_ids_in = arith.index_cast(T.index, i32_size_expert_ids_in)
        gx = (inter_in + fx.Index(int(tile_n) - 1)) // fx.Index(int(tile_n))
        gy = size_expert_ids_in
        launcher = moe_mxscale_stage1_single(
            arg_out, arg_x, arg_w, arg_scale_x, arg_scale_w,
            arg_sorted_token_ids, arg_expert_ids, arg_sorted_weights, arg_num_valid_ids,
            i32_tokens_in, i32_inter_in, i32_k_in, i32_size_expert_ids_in,
        )
        _cluster_arg = (int(cluster_m), int(cluster_n), 1) if use_cluster else None
        _finalize_alloc_and_launch_2d(
            ctx=ctx,
            alloc=alloc,
            launcher=launcher,
            gx=gx,
            gy=gy,
            block_threads=block_threads,
            stream=stream,
            waves_per_eu=effective_waves_per_eu,
            ir=ir,
            cluster=_cluster_arg,
            gz=int(k_batch),
        )

    if expert_sched_mode:
        launch_mxscale_stage1_single.compile_hints["llvm_options"] = {
            "amdgpu-expert-scheduling-mode": True,
        }

    return launch_mxscale_stage1_single


@functools.lru_cache(maxsize=64)
def _compile_stage2_mxscale_kernel_impl(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    route_tile_m: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    m_warp: int,
    n_warp: int,
    doweight_stage2: bool,
    out_dtype: str,
    accumulate: bool,
    waves_per_eu: int | None,
    data_format: str = "fp8",
    expert_sched_mode: bool = True,
    num_buffers: int = 1,
    use_tdm_gather: bool = True,
    use_tdm_gather_as: bool = True,
    use_tdm_store: bool = False,
    inst_prefetch: bool = False,
    wave_specialized_tdm: bool = False,
    cluster_m: int = 1,
    cluster_n: int = 1,
):
    """Compile mxscale stage2 single kernel (route-pack + TDM + WMMA_SCALE + epilog).

    ``use_tdm_gather_as`` enables the TDM-gather path for the A-scale matrix
    to eliminate the ``s_wait_dscnt`` stall cluster caused by per-byte
    ``buffer_load`` + ``ds_write_b8`` on the scalar A-scale path. Falls back
    to the vectorised scalar loader when the LDS scale layout is not row-major
    (``wmma_m_rep > 1`` and not ``is_fp4``) or the row width is below the TDM
    gather alignment (``scale_k_per_tile < 4`` / not a multiple of 4).
    """
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import llvm as llvm_dialect
    from flydsl._mlir.dialects import memref, scf
    from flydsl.compiler.kernel_function import CompilationContext
    from flydsl.expr import arith, buffer_ops, const_expr, gpu, idx2crd, range_constexpr, rocdl, tdm_ops, vector
    from flydsl.expr.typing import T
    from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr, get_op_result_or_value

    if bool(use_tdm_store) and bool(accumulate):
        raise ValueError("use_tdm_store is not compatible with accumulate=True in moe mxscale stage2")

    tp = _compute_mxscale_tiling(
        data_format=data_format, K=int(inter_dim),
        tile_m=int(tile_m), tile_n=int(tile_n), tile_k=int(tile_k),
        m_warp=int(m_warp), n_warp=int(n_warp), out_dtype=out_dtype,
        num_buffers=int(num_buffers), cluster_m=int(cluster_m),
        cluster_n=int(cluster_n), stage_name="stage2",
    )
    is_fp4, is_a8w4 = tp["is_fp4"], tp["is_a8w4"]
    PACK_FACTOR_A, PACK_FACTOR_B = tp["PACK_FACTOR_A"], tp["PACK_FACTOR_B"]
    ACC_VEC_SIZE = tp["ACC_VEC_SIZE"]
    DS_LOADS_PER_A_FRAG = tp["DS_LOADS_PER_A_FRAG"]
    WMMA_M, WMMA_N, WMMA_K = tp["WMMA_M"], tp["WMMA_N"], tp["WMMA_K"]
    SCALE_BLOCK, SCALES_PER_WMMA = tp["SCALE_BLOCK"], tp["SCALES_PER_WMMA"]
    WAVE_SIZE = tp["WAVE_SIZE"]
    LDS_PAD_A_BYTES, LDS_PAD_B_BYTES = tp["LDS_PAD_A_BYTES"], tp["LDS_PAD_B_BYTES"]
    use_cluster = tp["use_cluster"]
    K = tp["K"]
    K_packed_a, K_packed_b = tp["K_packed_a"], tp["K_packed_b"]
    packed_tile_k_a, packed_tile_k_b = tp["packed_tile_k_a"], tp["packed_tile_k_b"]
    K_scale, scale_k_per_tile = tp["K_scale"], tp["scale_k_per_tile"]
    block_threads = tp["block_threads"]
    warp_tile_m, warp_tile_n = tp["warp_tile_m"], tp["warp_tile_n"]
    wmma_m_rep, wmma_n_rep = tp["wmma_m_rep"], tp["wmma_n_rep"]
    k_wmma_steps, n_accs = tp["k_wmma_steps"], tp["n_accs"]
    num_k_tiles = tp["num_k_tiles"]
    b_scale_load_rep = tp["b_scale_load_rep"]
    interleaved_scale_cols_b = tp["interleaved_scale_cols_b"]
    lds_a_stride_bytes = tp["lds_a_stride_bytes"]
    lds_b_stride_bytes = tp["lds_b_stride_bytes"]
    lds_a_data_bytes, lds_b_data_bytes = tp["lds_a_data_bytes"], tp["lds_b_data_bytes"]
    lds_a_scale_bytes, lds_b_scale_bytes = tp["lds_a_scale_bytes"], tp["lds_b_scale_bytes"]
    interleaved_scale_cols_a = tp["interleaved_scale_cols_a"]

    N_total = int(model_dim)
    num_warps = int(m_warp) * int(n_warp)
    if bool(wave_specialized_tdm):
        if num_warps < 2:
            raise ValueError(
                f"wave_specialized_tdm requires at least 2 waves (B + B_scale), got {num_warps}")
    _tdm_loader_waves = 2
    tdm_desc_num_warps = 1 if bool(wave_specialized_tdm) else num_warps
    effective_waves_per_eu = waves_per_eu
    if use_cluster and effective_waves_per_eu is None:
        effective_waves_per_eu = 2

    # A-scale TDM gather gating mirrors stage1: requires A-side TDM gather
    # (for _a_tok_ids SGPRs), a row-major LDS scale layout, and a row width
    # that is a positive multiple of 4 bytes (TDM gather hardware constraint).
    _as_layout_rowmajor = bool(is_fp4) or (int(wmma_m_rep) == 1)
    _as_row_bytes_ok = int(scale_k_per_tile) >= 4 and (int(scale_k_per_tile) % 4 == 0)
    _use_tdm_gather_as = (
        bool(use_tdm_gather_as)
        and bool(use_tdm_gather)
        and _as_layout_rowmajor
        and _as_row_bytes_ok
    )

    _use_pipeline = int(num_buffers) >= 2
    if _use_pipeline:
        from kernels.gemm_common_gfx1250 import (
            pipeline_fence, pipeline_fence_signal, pipeline_fence_wait,
        )
        _B_TDM_PER_STEP = 1 if bool(wave_specialized_tdm) else 2
        _pp = _compute_pipeline_plan(
            num_k_tiles=num_k_tiles, num_buffers=int(num_buffers),
            B_TDM_PER_STEP=_B_TDM_PER_STEP, tile_m=int(tile_m),
            use_tdm_gather=use_tdm_gather,
            use_tdm_gather_as=_use_tdm_gather_as,
            wave_specialized_tdm=wave_specialized_tdm,
            tdm_loader_waves=_tdm_loader_waves,
        )
        pre_loaded = _pp["pre_loaded"]
        loop_iters = _pp["loop_iters"]
        _tail_start = _pp["tail_start"]
        extra = _pp["extra"]
        _A_GATHER_GROUPS = _pp["A_GATHER_GROUPS"]
        _AS_GATHER_GROUPS = _pp["AS_GATHER_GROUPS"]
        TDM_PER_STEP = _pp["TDM_PER_STEP"]
        _fence_outstanding = _pp["fence_outstanding"]
        _tail_plan = _pp["tail_plan"]
    from kernels.gemm_common_gfx1250 import workgroup_barrier

    alloc = SmemAllocator(
        None,
        arch=str(get_hip_arch()),
        global_sym_name=(
            f"moe_mxscale_{data_format}_s2_single_g{int(bool(use_tdm_gather))}"
            f"_as{int(_use_tdm_gather_as)}"
        ),
    )
    _nb = int(num_buffers)
    off_a_list, off_b_list, off_as_list, off_bs_list = [], [], [], []
    for _buf_i in range(_nb):
        _oa = alloc._align(alloc.ptr, 16)
        alloc.ptr = _oa + lds_a_data_bytes
        off_a_list.append(_oa)
        _ob = alloc._align(alloc.ptr, 16)
        alloc.ptr = _ob + lds_b_data_bytes
        off_b_list.append(_ob)
        _oas = alloc._align(alloc.ptr, 16)
        alloc.ptr = _oas + lds_a_scale_bytes
        off_as_list.append(_oas)
        _obs = alloc._align(alloc.ptr, 16)
        alloc.ptr = _obs + lds_b_scale_bytes
        off_bs_list.append(_obs)

    # lds_tid: preloaded sorted_token_ids for current M-tile (see stage1 comments).
    lds_tid_bytes = int(tile_m) * 4
    off_tid = alloc._align(alloc.ptr, 16)
    alloc.ptr = off_tid + lds_tid_bytes

    if bool(use_tdm_store):
        from kernels.gemm_common_gfx1250 import store_acc_vec8_to_lds
        _ds2 = _compute_tdm_store_layout(
            warp_tile_m=warp_tile_m, warp_tile_n=warp_tile_n,
            num_warps=num_warps, WMMA_N=WMMA_N, use_pipeline=_use_pipeline,
        )
        total_d_bytes = _ds2["total_d_bytes"]
        lds_d_row_stride = _ds2["lds_d_row_stride"]
        warp_d_bytes = _ds2["warp_d_bytes"]
        d_output_off = _ds2["d_output_off"]
        _lds_d_stride_elems = _ds2["lds_d_stride_elems"]
        _warp_d_elems = _ds2["warp_d_elems"]
        _n_col_d_elems = _ds2["n_col_d_elems"]
        d_need_epilogue_fence = _ds2["d_need_epilogue_fence"]
        elem_bytes_d = 2
        LDS_PAD_D_BYTES = 16
        if total_d_bytes > alloc.ptr:
            alloc.ptr = total_d_bytes

    _sub_tiles = _make_wmma_sub_tiles(
        wmma_m_rep=wmma_m_rep, wmma_n_rep=wmma_n_rep, WMMA_M=WMMA_M, is_fp4=is_fp4
    )

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def moe_mxscale_stage2_single(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_num_valid_ids: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_n_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
    ):
        _ = i32_k_in
        # ASTRewriter strips ``const_expr(...)`` from ``if`` tests, which would
        # otherwise eliminate every reference to ``const_expr`` from the
        # rewritten function body and shrink ``co_freevars`` by one — causing
        # CPython to reject ``f.__code__ = new_f_code_o`` because the original
        # ``__closure__`` length no longer matches. Keep one explicit reference
        # so the rewritten code object's free-vars list still includes
        # ``const_expr``.
        _keep_const_expr_ref = const_expr  # noqa: F841
        if const_expr(inst_prefetch):
            if arith.cmpi(arith.CmpIPredicate.eq, rocdl.wave_id(),
                          arith.constant(0, type=T.i32)):
                _prefetch_lines = ["s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 8, 1), 1"]
                for _pg in range_constexpr(10):
                    _prefetch_lines.append(
                        f"s_prefetch_inst_pc_rel {_pg * 4096}, s0, 31")
                llvm_dialect.inline_asm(
                    None, [],
                    "\n".join(_prefetch_lines),
                    "", has_side_effects=True,
                )
        llvm_dialect.inline_asm(
            None, [],
            "s_setreg_imm32_b32 hwreg(26, 4, 1), 1",
            "",
            has_side_effects=True,
        )

        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        by = gpu.block_id("y")

        tokens_idx = arith.index_cast(T.index, i32_tokens_in)
        n_idx = arith.index_cast(T.index, i32_n_in)
        size_expert_ids = arith.index_cast(T.index, i32_size_expert_ids_in)
        c_topk_i32 = arith.constant(int(topk), type=T.i32)
        num_valid_i32 = buffer_ops.buffer_load(
            buffer_ops.create_buffer_resource(arg_num_valid_ids, max_size=True),
            arith.constant(0, type=T.i32),
            vec_width=1,
            dtype=T.i32,
        )

        sorted_num = size_expert_ids * arith.index(int(route_tile_m))
        sorted_nbytes = sorted_num * arith.index(4)
        eid_nbytes = size_expert_ids * arith.index(4)
        x_rows = tokens_idx * arith.index(int(topk))
        x_nbytes = x_rows * arith.index(K_packed_a)
        sx_nbytes = x_rows * arith.index(K_scale)
        w_rows = arith.index(int(experts)) * n_idx
        w_nbytes = w_rows * arith.index(K_packed_b)
        sw_nbytes = w_rows * arith.index(K_scale)
        out_nbytes = tokens_idx * n_idx * arith.index(2)
        if const_expr(not bool(accumulate)):
            out_nbytes = x_rows * n_idx * arith.index(2)

        sorted_rsrc = buffer_ops.create_buffer_resource(arg_sorted_token_ids, max_size=False, num_records_bytes=sorted_nbytes)
        eid_rsrc = buffer_ops.create_buffer_resource(arg_expert_ids, max_size=False, num_records_bytes=eid_nbytes)
        x_rsrc = buffer_ops.create_buffer_resource(arg_x, max_size=False, num_records_bytes=x_nbytes)
        sx_rsrc = buffer_ops.create_buffer_resource(arg_scale_x, max_size=False, num_records_bytes=sx_nbytes)
        w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False, num_records_bytes=w_nbytes)
        sw_rsrc = buffer_ops.create_buffer_resource(arg_scale_w, max_size=False, num_records_bytes=sw_nbytes)
        out_rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=False, num_records_bytes=out_nbytes)
        tw_rsrc = buffer_ops.create_buffer_resource(arg_sorted_weights, max_size=True)

        eid_i32 = buffer_ops.buffer_load(eid_rsrc, arith.index_cast(T.i32, by), vec_width=1, dtype=T.i32)
        eid_ok0 = arith.cmpi(arith.CmpIPredicate.sge, eid_i32, arith.constant(0, type=T.i32))
        eid_ok1 = arith.cmpi(arith.CmpIPredicate.slt, eid_i32, arith.constant(int(experts), type=T.i32))
        block_row_start = arith.index_cast(T.i32, by * arith.index(int(route_tile_m)))
        block_in_valid = arith.cmpi(arith.CmpIPredicate.slt, block_row_start, num_valid_i32)
        block_ok = arith.andi(block_in_valid, arith.andi(eid_ok0, eid_ok1))

        layout_thr = _make_moe_wave_layout(m_warp=m_warp, n_warp=n_warp, WAVE_SIZE=WAVE_SIZE, fx=fx)
        thr_coord = idx2crd(tx, layout_thr)
        wave_m_idx, wave_n_idx, lane_kgrp, lane16 = (
            fx.get(thr_coord, 0), fx.get(thr_coord, 1), fx.get(thr_coord, 2), fx.get(thr_coord, 3)
        )
        warp_m_base = wave_m_idx * arith.index(warp_tile_m)
        warp_n_base = wave_n_idx * arith.index(warp_tile_n)
        blk_n = bx * arith.index(int(tile_n))

        if const_expr(use_cluster):
            _local_x, _local_y = gpu.compute_cluster_position()
            _a_mcast_mask, b_mcast_mask = gpu.compute_mcast_masks(
                _local_x, _local_y, int(cluster_m), int(cluster_n))
        else:
            b_mcast_mask = 0

        base_ptr = alloc.get_base()
        lds_a_bufs = []
        lds_b_bufs = []
        lds_as_bufs = []
        lds_bs_bufs = []
        for _bi in range_constexpr(_nb):
            _sa = SmemPtr(base_ptr, off_a_list[_bi], T.i8, shape=(lds_a_data_bytes,))
            _sb = SmemPtr(base_ptr, off_b_list[_bi], T.i8, shape=(lds_b_data_bytes,))
            _sas = SmemPtr(base_ptr, off_as_list[_bi], T.i8, shape=(lds_a_scale_bytes,))
            _sbs = SmemPtr(base_ptr, off_bs_list[_bi], T.i8, shape=(lds_b_scale_bytes,))
            lds_a_bufs.append(get_op_result_or_value(_sa.get()))
            lds_b_bufs.append(get_op_result_or_value(_sb.get()))
            lds_as_bufs.append(get_op_result_or_value(_sas.get()))
            lds_bs_bufs.append(get_op_result_or_value(_sbs.get()))

        lds_tid = SmemPtr(base_ptr, off_tid, T.i32, shape=(int(tile_m),)).get()

        if const_expr(bool(use_tdm_store)):
            from kernels.gemm_common_gfx1250 import get_lds_memref
            d_lds_f16_count = total_d_bytes // 2
            d_smem = SmemPtr(base_ptr, d_output_off, T.f16,
                             shape=(d_lds_f16_count,))
            d_lds_buffer = get_lds_memref(d_smem)
            warp_lds_off = (
                (wave_m_idx * arith.index(int(n_warp)) + wave_n_idx)
                * arith.index(_warp_d_elems)
            )
            d_lane_base = (
                warp_lds_off
                + lane16 * arith.index(_lds_d_stride_elems)
                + lane_kgrp * arith.index(4 * elem_bytes_d)
            )
            wave_id_idx = arith.index_cast(T.index, rocdl.wave_id())
            d_warp_off_sgpr = (
                wave_id_idx * arith.index(warp_d_bytes)
                + arith.index(d_output_off)
            )
            warp_m_off_sgpr = (
                (wave_id_idx / arith.index(int(n_warp)))
                * arith.index(warp_tile_m)
            )
            warp_n_off_sgpr = (
                (wave_id_idx % arith.index(int(n_warp)))
                * arith.index(warp_tile_n)
            )
            d_desc = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_out,
                lds_memref=base_ptr,
                global_offset=(
                    by * arith.index(int(tile_m)) + warp_m_off_sgpr,
                    blk_n + warp_n_off_sgpr,
                ),
                tensor_shape=(warp_tile_m, warp_tile_n),
                strides=(N_total, 1),
                tile_shape=(warp_tile_m, warp_tile_n),
                elem_bytes=elem_bytes_d,
                pad_interval=warp_tile_n,
                pad_amount=LDS_PAD_D_BYTES // elem_bytes_d,
                num_warps=1,
                lds_byte_offset=d_warp_off_sgpr,
                for_store=True,
            )

        _use_tdm_gather_a = bool(use_tdm_gather)
        _a_row_ids = []
        _a_row_valids = []
        _TDM_GATHER_CHUNK = 8
        _TDM_GATHER_GROUPS = (int(tile_m) + _TDM_GATHER_CHUNK - 1) // _TDM_GATHER_CHUNK
        _tokens_topk_sgpr = None

        def _sum_i32_values(_vals):
            _acc = arith.constant(0, type=T.i32)
            for _vi in range_constexpr(len(_vals)):
                _acc = _acc + _vals[_vi]
            return _acc

        def _get_tokens_topk_sgpr():
            nonlocal _tokens_topk_sgpr
            if const_expr(_tokens_topk_sgpr is None):
                _m_i32 = arith.index_cast(
                    T.i32,
                    tokens_idx * arith.index(int(topk)),
                )
                _tokens_topk_sgpr = rocdl.readfirstlane(T.i32, _m_i32)
            return _tokens_topk_sgpr

        def _preload_sorted_ids_to_lds():
            """Preload tile_m sorted_token_ids entries into ``lds_tid`` (once per CTA).

            See stage1 for the rationale. Invalid rows (row_in_route or
            row_in_valid false) are stored as sentinel ``0xFFFFFFFF`` so
            downstream ``tok_ok`` / ``slot_ok`` checks naturally reject them.
            """
            _tid_in_range = arith.cmpi(
                arith.CmpIPredicate.ult, tx, fx.Index(int(tile_m)))
            _if_tid = scf.IfOp(_tid_in_range)
            with ir.InsertionPoint(_if_tid.then_block):
                _tx_i32 = arith.index_cast(T.i32, tx)
                _sorted_row = by * fx.Index(int(tile_m)) + tx
                _sorted_i32 = arith.index_cast(T.i32, _sorted_row)
                _in_route = arith.cmpi(
                    arith.CmpIPredicate.ult,
                    _tx_i32,
                    arith.constant(int(route_tile_m), type=T.i32),
                )
                _in_valid = arith.cmpi(
                    arith.CmpIPredicate.slt, _sorted_i32, num_valid_i32)
                _row_valid = arith.andi(_in_route, _in_valid)
                _row_safe_i32 = arith.select(
                    _row_valid, _sorted_i32, block_row_start)
                _raw = buffer_ops.buffer_load(
                    sorted_rsrc, _row_safe_i32, vec_width=1, dtype=T.i32)
                _sentinel = arith.constant(-1, type=T.i32)  # 0xFFFFFFFF
                _val = arith.select(_row_valid, _raw, _sentinel)
                _vec1 = vector.from_elements(T.vec(1, T.i32), [_val])
                vector.store(_vec1, lds_tid, [tx], alignment=4)
                scf.YieldOp([])
            workgroup_barrier(use_cluster=use_cluster)

        def _load_fused_from_lds(row_index):
            if isinstance(row_index, int):
                row_index = arith.index(row_index)
            return memref.load(lds_tid, [row_index])

        def _precompute_a_row_indices():
            _safe_row = arith.constant(0, type=T.i32)
            _one_i32 = arith.constant(1, type=T.i32)
            _zero_i32 = arith.constant(0, type=T.i32)
            for _ri in range_constexpr(int(tile_m)):
                _fused = _load_fused_from_lds(_ri)
                _fused_sgpr = rocdl.readfirstlane(T.i32, _fused)
                _tok = _fused_sgpr & fx.Int32((1 << 24) - 1)
                _slot = _fused_sgpr >> fx.Int32(24)
                _tok_ok = arith.cmpi(arith.CmpIPredicate.ult, _tok, i32_tokens_in)
                _slot_ok0 = arith.cmpi(arith.CmpIPredicate.sge, _slot, fx.Int32(0))
                _slot_ok1 = arith.cmpi(arith.CmpIPredicate.slt, _slot, c_topk_i32)
                _ts = _tok * c_topk_i32 + _slot
                _ts_ok = arith.andi(_tok_ok, arith.andi(_slot_ok0, _slot_ok1))
                _row_fully_ok = _ts_ok
                _row_valid_i32 = arith.select(_row_fully_ok, _one_i32, _zero_i32)
                _a_row_valids.append(rocdl.readfirstlane(T.i32, _row_valid_i32))
                _ts_safe = arith.select(_row_fully_ok, _ts, _safe_row)
                _a_row_ids.append(rocdl.readfirstlane(T.i32, _ts_safe))

        def make_desc_a(k_base):
            return k_base / arith.index(PACK_FACTOR_A)

        def issue_a_load(k_packed_base, target_lds):
            total = int(tile_m * packed_tile_k_a)
            rounds = (total + block_threads - 1) // block_threads
            for it in range(rounds):
                elem = tx + fx.Index(it * block_threads)
                in_range = arith.cmpi(arith.CmpIPredicate.ult, arith.index_cast(T.i32, elem), arith.constant(total, type=T.i32))
                _if_elem = scf.IfOp(in_range)
                with ir.InsertionPoint(_if_elem.then_block):
                    row = elem // arith.index(int(packed_tile_k_a))
                    col = elem % arith.index(int(packed_tile_k_a))
                    # Use preloaded lds_tid instead of per-thread buffer_load(sorted_rsrc, ...).
                    fused = _load_fused_from_lds(row)
                    tok = fused & arith.constant((1 << 24) - 1, type=T.i32)
                    slot = fused >> arith.constant(24, type=T.i32)
                    tok_ok = arith.cmpi(arith.CmpIPredicate.ult, tok, i32_tokens_in)
                    slot_ok0 = arith.cmpi(arith.CmpIPredicate.sge, slot, arith.constant(0, type=T.i32))
                    slot_ok1 = arith.cmpi(arith.CmpIPredicate.slt, slot, c_topk_i32)
                    ts = tok * c_topk_i32 + slot
                    ts_ok = arith.andi(tok_ok, arith.andi(slot_ok0, slot_ok1))
                    load_ok = ts_ok
                    x_idx = ts * arith.constant(K_packed_a, type=T.i32) + arith.index_cast(T.i32, k_packed_base + col)
                    x_idx_safe = arith.select(load_ok, x_idx, arith.constant(0, type=T.i32))
                    x_val = arith.select(load_ok, buffer_ops.buffer_load(x_rsrc, x_idx_safe, vec_width=1, dtype=T.i8), arith.constant(0, type=T.i8))
                    lds_idx = row * arith.index(lds_a_stride_bytes) + col
                    v1 = vector.from_elements(T.vec(1, T.i8), [x_val])
                    vector.store(v1, target_lds, [lds_idx], alignment=1)
                    scf.YieldOp([])

        # Cache of K-invariant pieces of the stage2 TDM gather descriptor.
        # See the stage1 equivalent for the field-by-field rationale; the
        # only stage2 differences are using ``_a_row_ids`` /
        # ``_a_row_valids`` and ``_get_tokens_topk_sgpr`` (rows already encode
        # token*topk slots) instead of the stage1 row sources.
        _a_gather_cache = {}

        def _build_a_gather_base_descs(lds_bufs):
            if "desc" in _a_gather_cache:
                return
            _tokens_topk = _get_tokens_topk_sgpr()
            _zero_i32 = arith.constant(0, type=T.i32)
            _descs = []
            _preds = []
            _base_addr_lo = []
            for _gi in range_constexpr(_TDM_GATHER_GROUPS):
                _start = _gi * _TDM_GATHER_CHUNK
                _cnt = min(_TDM_GATHER_CHUNK, int(tile_m) - _start)
                _row_indices = _a_row_ids[_start:_start + _cnt]
                _valid_count = _sum_i32_values(_a_row_valids[_start:_start + _cnt])
                _has_valid = arith.cmpi(arith.CmpIPredicate.sgt, _valid_count, _zero_i32)
                _issue_pred = _has_valid
                if const_expr(wave_specialized_tdm):
                    _gather_owner = _gi % _tdm_loader_waves
                    _is_gather_loader = arith.cmpi(
                        arith.CmpIPredicate.eq,
                        _tdm_wave_id,
                        arith.constant(_gather_owner, type=T.i32),
                    )
                    _issue_pred = arith.andi(_issue_pred, _is_gather_loader)
                _preds.append(_issue_pred)

                _lds_off = fx.Index(_start * lds_a_stride_bytes)
                _per_buf = []
                # See stage1 note: range_constexpr is mandatory here so the
                # AST rewriter does not turn this into an scf.ForOp.
                for _buf_i in range_constexpr(len(lds_bufs)):
                    _base_desc = tdm_ops.make_tensor_gather_descriptor(
                        global_ptr=arg_x,
                        lds_memref=lds_bufs[_buf_i],
                        row_indices=_row_indices,
                        row_width=int(packed_tile_k_a),
                        tensor_dim0=K_packed_a,
                        tensor_dim1=_tokens_topk,
                        stride=K_packed_a,
                        elem_bytes=1,
                        pad_interval=int(packed_tile_k_a) if LDS_PAD_A_BYTES > 0 else 0,
                        pad_amount=LDS_PAD_A_BYTES if LDS_PAD_A_BYTES > 0 else 0,
                        index_size=32,
                        gather_tile_dim1=_valid_count,
                        lds_byte_offset=_lds_off,
                        global_byte_offset=None,
                    )
                    _per_buf.append(_base_desc)
                _descs.append(_per_buf)
                _base_addr_lo.append(vector.extract(
                    _per_buf[0].dgroup0,
                    static_position=[2],
                    dynamic_position=[],
                ))

            _a_gather_cache["desc"] = _descs
            _a_gather_cache["pred"] = _preds
            _a_gather_cache["base_addr_lo"] = _base_addr_lo

        def issue_a_load_tdm_gather(k_base, buf_idx):
            """Hot path: advance addr_lo on the precomputed gather descriptor.

            Requires ``_build_a_gather_base_descs(lds_bufs)`` to have been
            called once before the K loop with the matching LDS buffer list.
            """
            k_packed_base = k_base if PACK_FACTOR_A == 1 else k_base // fx.Index(PACK_FACTOR_A)
            _k_byte_off_i32 = arith.index_cast(T.i32, k_packed_base)
            _descs = _a_gather_cache["desc"]
            _preds = _a_gather_cache["pred"]
            _base_addr_lo = _a_gather_cache["base_addr_lo"]
            for _gi in range_constexpr(_TDM_GATHER_GROUPS):
                _if_issue = scf.IfOp(_preds[_gi])
                with ir.InsertionPoint(_if_issue.then_block):
                    _curr_addr_lo = arith.addi(
                        _base_addr_lo[_gi], _k_byte_off_i32)
                    tdm_ops.tensor_load_gather(
                        tdm_ops.update_tensor_gather_descriptor_addr_lo(
                            _descs[_gi][buf_idx],
                            _curr_addr_lo,
                        )
                    )
                    scf.YieldOp([])

        def make_desc_as(k_base):
            return k_base / arith.index(SCALE_BLOCK)

        def issue_as_load(k_scale_base, target_lds):
            """Vectorised scalar A-scale loader (Option B) for stage2.

            See stage1 ``issue_as_load`` for the 4-byte chunking rationale.
            Stage2 addresses rows by ``ts = tok * topk + slot`` and the
            ``load_ok`` guard checks both ``tok_ok`` and ``slot_ok``.
            """
            _blk_bytes = int(SCALES_PER_WMMA)
            _row_bytes = int(scale_k_per_tile)
            if const_expr(_row_bytes % _blk_bytes == 0 and _row_bytes >= _blk_bytes):
                _blk_vec_type = T.vec(_blk_bytes, T.i8)
                _blks_per_row = _row_bytes // _blk_bytes
                total = int(tile_m) * _blks_per_row
                rounds = (total + block_threads - 1) // block_threads
                for it in range(rounds):
                    elem = tx + fx.Index(it * block_threads)
                    in_range = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        arith.index_cast(T.i32, elem),
                        arith.constant(total, type=T.i32),
                    )
                    _if_elem = scf.IfOp(in_range)
                    with ir.InsertionPoint(_if_elem.then_block):
                        row = elem // arith.index(_blks_per_row)
                        ksc_blk = elem % arith.index(_blks_per_row)
                        fused = _load_fused_from_lds(row)
                        tok = fused & arith.constant((1 << 24) - 1, type=T.i32)
                        slot = fused >> arith.constant(24, type=T.i32)
                        tok_ok = arith.cmpi(
                            arith.CmpIPredicate.ult, tok, i32_tokens_in,
                        )
                        slot_ok0 = arith.cmpi(
                            arith.CmpIPredicate.sge, slot,
                            arith.constant(0, type=T.i32),
                        )
                        slot_ok1 = arith.cmpi(
                            arith.CmpIPredicate.slt, slot, c_topk_i32,
                        )
                        ts = tok * c_topk_i32 + slot
                        ts_ok = arith.andi(tok_ok, arith.andi(slot_ok0, slot_ok1))
                        if const_expr(_as_layout_rowmajor):
                            lds_idx = (
                                row * arith.index(_row_bytes)
                                + ksc_blk * arith.index(_blk_bytes)
                            )
                        else:
                            warp_row_idx = row / arith.index(warp_tile_m)
                            local_row = row % arith.index(warp_tile_m)
                            lane_row = local_row % arith.index(WMMA_M)
                            local_wm_idx = local_row / arith.index(WMMA_M)
                            global_lds_row = (
                                warp_row_idx * arith.index(WMMA_M) + lane_row
                            )
                            lds_idx = (
                                global_lds_row
                                * arith.index(interleaved_scale_cols_a)
                                + ksc_blk
                                * arith.index(wmma_m_rep * SCALES_PER_WMMA)
                                + local_wm_idx * arith.index(SCALES_PER_WMMA)
                            )
                        _if_ok = scf.IfOp(ts_ok, has_else=True)
                        with ir.InsertionPoint(_if_ok.then_block):
                            chunk_off = (
                                k_scale_base
                                + ksc_blk * arith.index(_blk_bytes)
                            )
                            sx_idx = (
                                ts * arith.constant(K_scale, type=T.i32)
                                + arith.index_cast(T.i32, chunk_off)
                            )
                            sx_raw = buffer_ops.buffer_load(
                                sx_rsrc,
                                arith.shrui(
                                    sx_idx,
                                    arith.constant(2, type=T.i32),
                                ),
                                vec_width=1,
                                dtype=T.i32,
                            )
                            sx_vec = vector.bitcast(
                                _blk_vec_type,
                                vector.from_elements(T.vec(1, T.i32), [sx_raw]),
                            )
                            vector.store(
                                sx_vec, target_lds, [lds_idx],
                                alignment=_blk_bytes,
                            )
                            scf.YieldOp([])
                        with ir.InsertionPoint(_if_ok.else_block):
                            fill_vec = vector.bitcast(
                                _blk_vec_type,
                                vector.from_elements(
                                    T.vec(1, T.i32),
                                    [arith.constant(0x7F7F7F7F, type=T.i32)],
                                ),
                            )
                            vector.store(
                                fill_vec, target_lds, [lds_idx],
                                alignment=_blk_bytes,
                            )
                            scf.YieldOp([])
                        scf.YieldOp([])
            else:
                total = int(tile_m * scale_k_per_tile)
                rounds = (total + block_threads - 1) // block_threads
                for it in range(rounds):
                    elem = tx + fx.Index(it * block_threads)
                    in_range = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        arith.index_cast(T.i32, elem),
                        arith.constant(total, type=T.i32),
                    )
                    _if_elem = scf.IfOp(in_range)
                    with ir.InsertionPoint(_if_elem.then_block):
                        row = elem // arith.index(int(scale_k_per_tile))
                        ksc = elem % arith.index(int(scale_k_per_tile))
                        fused = _load_fused_from_lds(row)
                        tok = fused & arith.constant((1 << 24) - 1, type=T.i32)
                        slot = fused >> arith.constant(24, type=T.i32)
                        tok_ok = arith.cmpi(arith.CmpIPredicate.ult, tok, i32_tokens_in)
                        slot_ok0 = arith.cmpi(arith.CmpIPredicate.sge, slot, arith.constant(0, type=T.i32))
                        slot_ok1 = arith.cmpi(arith.CmpIPredicate.slt, slot, c_topk_i32)
                        ts = tok * c_topk_i32 + slot
                        ts_ok = arith.andi(tok_ok, arith.andi(slot_ok0, slot_ok1))
                        load_ok = ts_ok
                        ksc_off = k_scale_base + ksc
                        sx_idx = ts * arith.constant(K_scale, type=T.i32) + arith.index_cast(T.i32, ksc_off)
                        sx_idx_safe = arith.select(load_ok, sx_idx, arith.constant(0, type=T.i32))
                        sx_val = arith.select(
                            load_ok,
                            buffer_ops.buffer_load(sx_rsrc, sx_idx_safe, vec_width=1, dtype=T.i8),
                            arith.constant(127, type=T.i8),
                        )
                        if is_fp4:
                            lds_idx = row * arith.index(int(scale_k_per_tile)) + ksc
                        else:
                            warp_row_idx = row / arith.index(warp_tile_m)
                            local_row = row % arith.index(warp_tile_m)
                            lane_row = local_row % arith.index(WMMA_M)
                            local_wm_idx = local_row / arith.index(WMMA_M)
                            global_lds_row = warp_row_idx * arith.index(WMMA_M) + lane_row
                            ksc_blk = ksc / arith.index(SCALES_PER_WMMA)
                            ksc_sub = ksc % arith.index(SCALES_PER_WMMA)
                            lds_idx = (
                                global_lds_row * arith.index(interleaved_scale_cols_a)
                                + ksc_blk * arith.index(wmma_m_rep * SCALES_PER_WMMA)
                                + local_wm_idx * arith.index(SCALES_PER_WMMA)
                                + ksc_sub
                            )
                        v1 = vector.from_elements(T.vec(1, T.i8), [sx_val])
                        vector.store(v1, target_lds, [lds_idx], alignment=1)
                        scf.YieldOp([])

        def issue_as_load_tdm_gather(k_scale_base, target_lds):
            """TDM-gather A-scale loader (Option A) for stage2.

            Reuses the ``_a_row_ids`` / ``_a_row_valids`` SGPR caches populated
            by ``_precompute_a_row_indices()`` for the stage2 A-data TDM path,
            where each row index is ``ts = tok * topk + slot``. Routes
            completion through ``tdm_cnt`` to eliminate the ``s_wait_dscnt 0``
            stall cluster caused by per-byte ``buffer_load`` + ``ds_write_b8``.

            Pre-conditions (enforced by the gating in this kernel):
              - ``use_tdm_gather=True`` (otherwise ``_a_row_ids`` is empty).
              - Row-major LDS scale layout (``is_fp4`` or ``wmma_m_rep == 1``).
              - ``scale_k_per_tile`` is a positive multiple of 4.
            """
            _as_row_bytes = int(scale_k_per_tile)
            _tokens_topk = _get_tokens_topk_sgpr()
            _zero_i32 = arith.constant(0, type=T.i32)
            for _gi in range_constexpr(_TDM_GATHER_GROUPS):
                _start = _gi * _TDM_GATHER_CHUNK
                _cnt = min(_TDM_GATHER_CHUNK, int(tile_m) - _start)
                _row_indices = _a_row_ids[_start:_start + _cnt]
                _valid_count = _sum_i32_values(_a_row_valids[_start:_start + _cnt])
                _lds_off = fx.Index(_start * _as_row_bytes)
                _has_valid = arith.cmpi(
                    arith.CmpIPredicate.sgt, _valid_count, _zero_i32,
                )
                _issue_pred = _has_valid
                if wave_specialized_tdm:
                    _gather_owner = _gi % _tdm_loader_waves
                    _is_gather_loader = arith.cmpi(
                        arith.CmpIPredicate.eq,
                        _tdm_wave_id,
                        arith.constant(_gather_owner, type=T.i32),
                    )
                    _issue_pred = arith.andi(_issue_pred, _is_gather_loader)
                _if_issue = scf.IfOp(_issue_pred)
                with ir.InsertionPoint(_if_issue.then_block):
                    desc = tdm_ops.make_tensor_gather_descriptor(
                        global_ptr=arg_scale_x,
                        lds_memref=target_lds,
                        row_indices=_row_indices,
                        row_width=_as_row_bytes,
                        tensor_dim0=int(K_scale),
                        tensor_dim1=_tokens_topk,
                        stride=int(K_scale),
                        elem_bytes=1,
                        pad_interval=0,
                        pad_amount=0,
                        index_size=32,
                        gather_tile_dim1=_valid_count,
                        lds_byte_offset=_lds_off,
                        global_byte_offset=k_scale_base,
                    )
                    tdm_ops.tensor_load_gather(desc)
                    scf.YieldOp([])

        def make_desc_b(n_off, k_base, target_lds):
            if const_expr(is_fp4):
                return tdm_ops.make_tensor_descriptor_2d(
                    global_ptr=arg_w, lds_memref=target_lds,
                    global_offset=(n_off, k_base / arith.index(PACK_FACTOR_B)),
                    tensor_shape=(int(tile_n), int(packed_tile_k_b)),
                    strides=(K_packed_b, 1),
                    tile_shape=(int(tile_n), int(packed_tile_k_b)),
                    elem_bytes=1, pad_interval=int(packed_tile_k_b), pad_amount=LDS_PAD_B_BYTES,
                    num_warps=tdm_desc_num_warps, workgroup_mask=b_mcast_mask)
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_w, lds_memref=target_lds,
                global_offset=(n_off / arith.index(16), (k_base / arith.index(PACK_FACTOR_B)) * arith.index(16)),
                tensor_shape=(int(N_total // 16), int(K_packed_b * 16)),
                strides=(int(K_packed_b * 16), 1),
                tile_shape=(int(tile_n // 16), int(packed_tile_k_b * 16)),
                elem_bytes=1,
                pad_interval=0, pad_amount=0,
                num_warps=tdm_desc_num_warps,
                workgroup_mask=b_mcast_mask)

        def make_desc_bs(n_off, k_base, target_lds):
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_scale_w, lds_memref=target_lds,
                global_offset=(n_off, k_base / arith.index(SCALE_BLOCK)),
                tensor_shape=(int(tile_n), int(scale_k_per_tile)),
                strides=(K_scale, 1),
                tile_shape=(int(tile_n), int(scale_k_per_tile)),
                elem_bytes=1, pad_interval=0, pad_amount=0,
                num_warps=tdm_desc_num_warps, workgroup_mask=b_mcast_mask)

        # Cache of K-invariant 2D B / B-scale descriptors used by stage2's
        # ``_issue_b_tdm_only``. Stage2 has no merge_gate_up_tdm path, so the
        # cache is single-branched. Mirrors the stage1 helper pair; closed
        # over ``make_desc_b``, ``make_desc_bs``, ``lds_b_bufs``,
        # ``lds_bs_bufs`` and ``eid_i32`` / ``n_idx`` / ``blk_n``, all
        # resolved at call time inside ``_if_blk``.
        _b_desc_cache = {}

        def _extract_desc_addr_lo(desc):
            return vector.extract(
                desc.dgroup0,
                static_position=[2],
                dynamic_position=[],
            )

        def _build_b_base_descs():
            if "ready" in _b_desc_cache:
                return
            _zero_k = arith.index(0)
            _eid_idx = arith.index_cast(T.index, eid_i32)
            _n_off = _eid_idx * n_idx + blk_n
            _b = [
                make_desc_b(_n_off, _zero_k, lds_b_bufs[i])
                for i in range_constexpr(_nb)
            ]
            _bs = [
                make_desc_bs(_n_off, _zero_k, lds_bs_bufs[i])
                for i in range_constexpr(_nb)
            ]
            _b_desc_cache["b"] = _b
            _b_desc_cache["bs"] = _bs
            _b_desc_cache["b_addr_lo"] = [_extract_desc_addr_lo(d) for d in _b]
            _b_desc_cache["bs_addr_lo"] = [_extract_desc_addr_lo(d) for d in _bs]
            _b_desc_cache["ready"] = True

        def _b_data_k_byte_off(k_base):
            # Fastest-axis byte offset for stage2 B data descriptor:
            #   non-fp4 : (k_base / PACK_FACTOR_B) * 16 bytes
            #   fp4     : (k_base / PACK_FACTOR_B) bytes
            # Matches make_desc_b global_offset math (elem_bytes=1).
            _k_packed_b = (
                k_base if PACK_FACTOR_B == 1
                else k_base // fx.Index(PACK_FACTOR_B)
            )
            if const_expr(is_fp4):
                return arith.index_cast(T.i32, _k_packed_b)
            return arith.index_cast(
                T.i32, _k_packed_b * fx.Index(16))

        def _b_scale_k_byte_off(k_base):
            return arith.index_cast(
                T.i32, k_base // fx.Index(SCALE_BLOCK))

        def issue_b_load(k_base, target_lds_b, target_lds_bs):
            eid_idx = arith.index_cast(T.index, eid_i32)
            n_off = eid_idx * n_idx + blk_n
            tdm_ops.tensor_load_2d(make_desc_b(n_off, k_base, target_lds_b))
            tdm_ops.tensor_load_2d(make_desc_bs(n_off, k_base, target_lds_bs))

        _ldrs = _make_mxscale_data_loaders(
            tiling=tp, warp_m_base=warp_m_base, warp_n_base=warp_n_base,
            wave_n_idx=wave_n_idx, lane16=lane16, lane_kgrp=lane_kgrp,
            ir=ir, arith=arith, vector=vector, llvm_dialect=llvm_dialect,
            T=T, range_constexpr=range_constexpr,
        )
        _lds_load_b128 = _ldrs["_lds_load_b128"]
        load_data_frag = _ldrs["load_data_frag"]
        load_b_frag = _ldrs["load_b_frag"]
        load_scale_i32 = _ldrs["load_scale_i32"]
        _precompute_a_data_bases = _ldrs["_precompute_a_data_bases"]
        _precompute_b_data_bases = _ldrs["_precompute_b_data_bases"]
        _precompute_a_scale_lane_bases = _ldrs["_precompute_a_scale_lane_bases"]
        _precompute_b_scale_lane_bases = _ldrs["_precompute_b_scale_lane_bases"]
        load_scale_b128 = _ldrs["load_scale_b128"]

        acc_zero = arith.constant_vector(0.0, T.vec(ACC_VEC_SIZE, T.f32))
        acc = [acc_zero] * n_accs

        _if_blk = scf.IfOp(block_ok)
        with ir.InsertionPoint(_if_blk.then_block):
            _preload_sorted_ids_to_lds()
            if const_expr(_use_tdm_gather_a):
                _precompute_a_row_indices()
            a_data_bases = _precompute_a_data_bases()
            b_data_bases = _precompute_b_data_bases()
            as_bases = _precompute_a_scale_lane_bases()
            bs_bases = _precompute_b_scale_lane_bases()
            _use_scheduled_compute = _use_pipeline and not is_fp4
            _front_wm = (wmma_m_rep + 1) // 2
            _back_wm = wmma_m_rep - _front_wm
            _front_wmma = _front_wm * wmma_n_rep
            _back_wmma = _back_wm * wmma_n_rep
            _b_frag_ds_loads_per_wn = 2 if is_a8w4 else 4
            _a_scale_ds_loads = wmma_m_rep if is_fp4 else (wmma_m_rep + 3) // 4
            _b_scale_ds_loads = b_scale_load_rep if is_fp4 else wmma_n_rep
            _bs_ds_loads = (
                wmma_n_rep * _b_frag_ds_loads_per_wn
                + _b_scale_ds_loads
                + _a_scale_ds_loads
            )

            # ── compute-tile helper ──────────────────────────────────
            def emit_wmma(accs, wm, wn, a_frag, b_frags, a_scales, b_scales):
                _mxscale_emit_wmma(
                    accs=accs, wm=wm, wn=wn,
                    a_frag=a_frag, b_frags=b_frags,
                    a_scales=a_scales, b_scales=b_scales,
                    is_fp4=is_fp4, is_a8w4=is_a8w4,
                    use_scale_opsel=False,
                    rocdl=rocdl, T=T,
                )

            def _compute_k_tile(accs_in, buf_idx, mid_compute_callback=None):
                _mid_emit_ks = 0
                if const_expr(k_wmma_steps > 1):
                    _mid_emit_wm = wmma_m_rep - 1
                    _mid_emit_wn = wmma_n_rep - 1
                else:
                    _front_wm = (wmma_m_rep + 1) // 2
                    _front_wn = (wmma_n_rep + 1) // 2
                    if const_expr(wmma_m_rep > 1):
                        _mid_emit_wm = _front_wm - 1
                        _mid_emit_wn = wmma_n_rep - 1
                    else:
                        _mid_emit_wm = 0
                        _mid_emit_wn = _front_wn - 1
                _did_mid = False
                for ks in range_constexpr(k_wmma_steps):
                    b_v = [load_b_frag(lds_b_bufs[buf_idx], b_data_bases, wn, ks)
                           for wn in range_constexpr(wmma_n_rep)]
                    if const_expr(is_fp4):
                        as_v = [load_scale_i32(lds_as_bufs[buf_idx], as_bases[wm], ks)
                                for wm in range_constexpr(wmma_m_rep)]
                        bs_v = [load_scale_i32(lds_bs_bufs[buf_idx], bs_bases[bi], ks)
                                for bi in range_constexpr(b_scale_load_rep)]
                    else:
                        as_v = load_scale_b128(lds_as_bufs[buf_idx], as_bases[0],
                                               wmma_m_rep, ks)
                        bs_v = [load_scale_i32(lds_bs_bufs[buf_idx], bs_bases[wn], ks)
                                for wn in range_constexpr(wmma_n_rep)]
                    for wm in range_constexpr(wmma_m_rep):
                        a_frag = load_data_frag(lds_a_bufs[buf_idx],
                                                a_data_bases[wm], ks)
                        for wn in range_constexpr(wmma_n_rep):
                            emit_wmma(accs_in, wm, wn, a_frag, b_v, as_v, bs_v)
                            if const_expr(
                                not _did_mid
                                and mid_compute_callback is not None
                                and ks == _mid_emit_ks
                                and wm == _mid_emit_wm
                                and wn == _mid_emit_wn
                            ):
                                mid_compute_callback()
                                _did_mid = True
                return accs_in

            def _load_b_and_scales(buf_idx, ks):
                b_v = [load_b_frag(lds_b_bufs[buf_idx], b_data_bases, wn, ks)
                       for wn in range_constexpr(wmma_n_rep)]
                if const_expr(is_fp4):
                    as_v = [load_scale_i32(lds_as_bufs[buf_idx], as_bases[wm], ks)
                            for wm in range_constexpr(wmma_m_rep)]
                    bs_v = [load_scale_i32(lds_bs_bufs[buf_idx], bs_bases[bi], ks)
                            for bi in range_constexpr(b_scale_load_rep)]
                else:
                    as_v = load_scale_b128(lds_as_bufs[buf_idx], as_bases[0],
                                           wmma_m_rep, ks)
                    bs_v = [load_scale_i32(lds_bs_bufs[buf_idx], bs_bases[wn], ks)
                            for wn in range_constexpr(wmma_n_rep)]
                return b_v, bs_v, as_v

            def _emit_rows(accs_in, start_wm, a_frags, b_frags, a_scales, b_scales):
                for frag_i in range_constexpr(len(a_frags)):
                    wm = start_wm + frag_i
                    for wn_raw in range_constexpr(wmma_n_rep):
                        wn = (wmma_n_rep - 1 - wn_raw) if (wm % 2 == 1) else wn_raw
                        emit_wmma(accs_in, wm, wn, a_frags[frag_i], b_frags, a_scales, b_scales)

            def _a_streaming_compute(
                accs_in,
                buf_idx,
                b_frags,
                b_scales,
                a_scales,
                ks,
                next_bs_info=None,
                mid_compute_callback=None,
            ):
                current_accs = accs_in
                next_result = None
                a_frags_front = [
                    load_data_frag(lds_a_bufs[buf_idx], a_data_bases[wm], ks)
                    for wm in range_constexpr(_front_wm)
                ]
                _use_partial_drain = (
                    next_bs_info is not None
                    and _front_wm * wmma_n_rep >= 4
                )

                if const_expr(_use_partial_drain):
                    _next_buf_idx, _next_ks = next_bs_info
                    next_result = _load_b_and_scales(_next_buf_idx, _next_ks)
                    rocdl.s_wait_dscnt(_bs_ds_loads)
                else:
                    rocdl.s_wait_dscnt(0)

                _emit_rows(current_accs, 0, a_frags_front, b_frags, a_scales, b_scales)

                if const_expr(mid_compute_callback is not None):
                    rocdl.sched_barrier(0)
                    mid_compute_callback()

                if const_expr(_back_wm > 0):
                    a_frags_back = [
                        load_data_frag(
                            lds_a_bufs[buf_idx],
                            a_data_bases[_front_wm + h],
                            ks,
                        )
                        for h in range_constexpr(_back_wm)
                    ]
                    _back_drain = _bs_ds_loads if _use_partial_drain else 0
                    rocdl.s_wait_dscnt(_back_drain)
                    _emit_rows(
                        current_accs,
                        _front_wm,
                        a_frags_back,
                        b_frags,
                        a_scales,
                        b_scales,
                    )

                if const_expr(_use_partial_drain):
                    return current_accs, next_result
                if const_expr(next_bs_info is not None):
                    _next_buf_idx, _next_ks = next_bs_info
                    next_result = _load_b_and_scales(_next_buf_idx, _next_ks)
                    return current_accs, next_result
                return current_accs

            def _compute_k_tile_scheduled(accs_in, buf_idx, mid_compute_callback=None):
                current_accs = list(accs_in)
                if const_expr(k_wmma_steps == 1):
                    b_v, bs_v, as_v = _load_b_and_scales(buf_idx, 0)
                    current_accs = _a_streaming_compute(
                        current_accs,
                        buf_idx,
                        b_v,
                        bs_v,
                        as_v,
                        0,
                        mid_compute_callback=mid_compute_callback,
                    )
                else:
                    prev_b, prev_bs, prev_as = _load_b_and_scales(buf_idx, 0)
                    for ks in range_constexpr(k_wmma_steps - 1):
                        _mid_cb = mid_compute_callback if ks == 0 else None
                        current_accs, (prev_b, prev_bs, prev_as) = _a_streaming_compute(
                            current_accs,
                            buf_idx,
                            prev_b,
                            prev_bs,
                            prev_as,
                            ks,
                            next_bs_info=(buf_idx, ks + 1),
                            mid_compute_callback=_mid_cb,
                        )
                    current_accs = _a_streaming_compute(
                        current_accs,
                        buf_idx,
                        prev_b,
                        prev_bs,
                        prev_as,
                        k_wmma_steps - 1,
                    )
                return current_accs

            def _hot_loop_scheduler_scheduled():
                if const_expr(not _use_scheduled_compute):
                    return
                _front_a_loads = _front_wm * DS_LOADS_PER_A_FRAG
                _back_a_loads = _back_wm * DS_LOADS_PER_A_FRAG
                for _ks in range_constexpr(k_wmma_steps):
                    if const_expr(_ks == 0):
                        rocdl.sched_dsrd(_bs_ds_loads + _front_a_loads)
                    else:
                        rocdl.sched_dsrd(_front_a_loads)
                    rocdl.sched_mfma(_front_wmma)
                    if const_expr(_back_wmma > 0):
                        rocdl.sched_dsrd(_back_a_loads)
                        rocdl.sched_mfma(_back_wmma)
                    if const_expr(_ks < k_wmma_steps - 1):
                        rocdl.sched_dsrd(_bs_ds_loads)
                rocdl.sched_barrier(0)

            if const_expr(wave_specialized_tdm):
                _tdm_wave_id = rocdl.wave_id()
                _is_loader_wave = arith.cmpi(
                    arith.CmpIPredicate.ult,
                    _tdm_wave_id,
                    arith.constant(_tdm_loader_waves, type=T.i32),
                )
                _tdm_pred = arith.constant(1, type=T.i32)

                def _select_wave_tdm_value(b_value, bs_value):
                    _wave_is_b = arith.cmpi(
                        arith.CmpIPredicate.eq,
                        _tdm_wave_id,
                        arith.constant(0, type=T.i32),
                    )
                    return arith.select(_wave_is_b, b_value, bs_value)

                def _tdm_desc_lds_addr(desc):
                    return vector.extract(
                        desc.dgroup0,
                        static_position=[1],
                        dynamic_position=[],
                    )

                def _tdm_desc_addr_lo(desc):
                    return vector.extract(
                        desc.dgroup0,
                        static_position=[2],
                        dynamic_position=[],
                    )

                def _tdm_desc_addr_hi(desc):
                    return vector.extract(
                        desc.dgroup0,
                        static_position=[3],
                        dynamic_position=[],
                    )

                _eid = arith.index_cast(T.index, eid_i32)
                _n_init = _eid * n_idx + blk_n
                _zero_k_base = arith.index(0)
                _data_adv_i32 = arith.constant(
                    packed_tile_k_b if is_fp4 else packed_tile_k_b * 16,
                    type=T.i32,
                )
                _scale_adv_i32 = arith.constant(scale_k_per_tile, type=T.i32)

                _stages_b_lds_addr = [
                    _tdm_desc_lds_addr(
                        make_desc_b(
                            _n_init,
                            _zero_k_base,
                            lds_b_bufs[i],
                        )
                    )
                    for i in range_constexpr(_nb)
                ]
                _stages_bs_lds_addr = [
                    _tdm_desc_lds_addr(
                        make_desc_bs(
                            _n_init,
                            _zero_k_base,
                            lds_bs_bufs[i],
                        )
                    )
                    for i in range_constexpr(_nb)
                ]

                _desc_b_init = make_desc_b(
                    _n_init,
                    _zero_k_base,
                    lds_b_bufs[0],
                )
                _desc_bs_init = make_desc_bs(
                    _n_init,
                    _zero_k_base,
                    lds_bs_bufs[0],
                )

                _active_stage_lds_addr = [
                    _select_wave_tdm_value(
                        _stages_b_lds_addr[i],
                        _stages_bs_lds_addr[i],
                    )
                    for i in range_constexpr(_nb)
                ]
                _active_addr_lo = _select_wave_tdm_value(
                    _tdm_desc_addr_lo(_desc_b_init),
                    _tdm_desc_addr_lo(_desc_bs_init),
                )
                _active_addr_hi = _select_wave_tdm_value(
                    _tdm_desc_addr_hi(_desc_b_init),
                    _tdm_desc_addr_hi(_desc_bs_init),
                )
                _active_dgroup1 = _select_wave_tdm_value(
                    _desc_b_init.dgroup1,
                    _desc_bs_init.dgroup1,
                )
                _active_adv_i32 = _select_wave_tdm_value(
                    _data_adv_i32,
                    _scale_adv_i32,
                )

                # See stage1 for rationale: pre-build per-stage TDMDescriptor2D
                # bases so the hot path only patches lane 2 via
                # update_tensor_descriptor_2d_addr_lo.
                _tdm_zero_addr_lo = arith.constant(0, type=T.i32)
                _active_stage_desc_base = [
                    tdm_ops.TDMDescriptor2D(
                        vector.from_elements(T.vec(4, T.i32), [
                            _tdm_pred,
                            _active_stage_lds_addr[i],
                            _tdm_zero_addr_lo,
                            _active_addr_hi,
                        ]),
                        _active_dgroup1,
                    )
                    for i in range_constexpr(_nb)
                ]

                def _issue_active_b_tdm_only(stage_idx, curr_addr_lo):
                    _if_loader = scf.IfOp(_is_loader_wave)
                    with ir.InsertionPoint(_if_loader.then_block):
                        tdm_ops.tensor_load_2d(
                            tdm_ops.update_tensor_descriptor_2d_addr_lo(
                                _active_stage_desc_base[stage_idx],
                                curr_addr_lo,
                            )
                        )
                        scf.YieldOp([])
                    _next_addr_lo = arith.addi(curr_addr_lo, _active_adv_i32)
                    return arith.select(
                        _is_loader_wave,
                        _next_addr_lo,
                        curr_addr_lo,
                    )

            if const_expr(_use_tdm_gather_a):
                _build_a_gather_base_descs(lds_a_bufs)
            # See stage1 for the rationale of guarding on wave_specialized_tdm:
            # in wave-specialized mode the hot path goes through
            # ``_issue_active_b_tdm_only``; ``_issue_b_tdm_only`` is only used
            # in non-pipelined / tail paths, so skip the cache build to avoid
            # emitting dead IR.
            if const_expr(not wave_specialized_tdm):
                _build_b_base_descs()

            # ── pipeline load helpers ─────────────────────────────────
            def _issue_b_tdm_only(k_base, buf_idx):
                _k_data_off = _b_data_k_byte_off(k_base)
                _k_scale_off = _b_scale_k_byte_off(k_base)
                _b_addr = arith.addi(
                    _b_desc_cache["b_addr_lo"][buf_idx], _k_data_off)
                _bs_addr = arith.addi(
                    _b_desc_cache["bs_addr_lo"][buf_idx], _k_scale_off)
                tdm_ops.tensor_load_2d(
                    tdm_ops.update_tensor_descriptor_2d_addr_lo(
                        _b_desc_cache["b"][buf_idx], _b_addr))
                tdm_ops.tensor_load_2d(
                    tdm_ops.update_tensor_descriptor_2d_addr_lo(
                        _b_desc_cache["bs"][buf_idx], _bs_addr))

            def _issue_scalar_loads(k_base, buf_idx):
                if const_expr(_use_tdm_gather_a):
                    issue_a_load_tdm_gather(k_base, buf_idx)
                else:
                    issue_a_load(make_desc_a(k_base), lds_a_bufs[buf_idx])
                if _use_tdm_gather_as:
                    issue_as_load_tdm_gather(make_desc_as(k_base), lds_as_bufs[buf_idx])
                else:
                    issue_as_load(make_desc_as(k_base), lds_as_bufs[buf_idx])

            def _issue_all_loads(k_base, buf_idx):
                _issue_b_tdm_only(k_base, buf_idx)
                _issue_scalar_loads(k_base, buf_idx)

            def _compute_with_mid_loads(accs_in, buf_idx, mid_load_callback=None):
                if const_expr(_use_scheduled_compute):
                    return _compute_k_tile_scheduled(
                        accs_in, buf_idx,
                        mid_compute_callback=mid_load_callback,
                    )
                return _compute_k_tile(
                    accs_in, buf_idx,
                    mid_compute_callback=mid_load_callback,
                )

            # ── main K-dimension reduction ────────────────────────────
            if const_expr(not _use_pipeline):
                # Single-buffer path (num_buffers=1)
                if const_expr(wave_specialized_tdm):
                    active_b_addr_lo = _active_addr_lo
                    for kt in range_constexpr(num_k_tiles):
                        k_base = fx.Index(kt * int(tile_k))
                        active_b_addr_lo = _issue_active_b_tdm_only(
                            0, active_b_addr_lo)
                        _issue_scalar_loads(k_base, 0)
                        tdm_ops.tensor_wait(0)
                        workgroup_barrier(use_cluster=use_cluster)
                        acc = _compute_k_tile(acc, 0)
                        workgroup_barrier(use_cluster=use_cluster)
                else:
                    for kt in range_constexpr(num_k_tiles):
                        k_base = fx.Index(kt * int(tile_k))
                        _issue_all_loads(k_base, 0)
                        tdm_ops.tensor_wait(0)
                        workgroup_barrier(use_cluster=use_cluster)
                        acc = _compute_k_tile(acc, 0)
                        workgroup_barrier(use_cluster=use_cluster)
            else:
                # Multi-buffer pipeline
                # ── prologue: pre-load first `pre_loaded` stages ──
                if const_expr(wave_specialized_tdm):
                    active_b_addr_lo = _active_addr_lo
                    for _pi in range_constexpr(pre_loaded):
                        active_b_addr_lo = _issue_active_b_tdm_only(
                            _pi, active_b_addr_lo)
                        _issue_scalar_loads(fx.Index(_pi * int(tile_k)), _pi)
                else:
                    for _pi in range_constexpr(pre_loaded):
                        _issue_all_loads(fx.Index(_pi * int(tile_k)), _pi)
                pipeline_fence(outstanding=0, use_cluster=use_cluster)

                # ── main pipelined loop ──
                if const_expr(loop_iters > 0):
                    if const_expr(wave_specialized_tdm):
                        _init = list(acc) + [active_b_addr_lo]
                        for _li, _st in fx.range(0, loop_iters, 1, init=_init):
                            _acc = list(_st[:n_accs])
                            _cur_b_addr_lo = _st[n_accs]
                            for _bi in range_constexpr(_nb):
                                _lb = (_bi + _nb - 1) % _nb
                                _kt = (_li * fx.Index(_nb)
                                       + fx.Index(pre_loaded + _bi))
                                _kb = _kt * fx.Index(int(tile_k))
                                pipeline_fence_signal(
                                    outstanding=_fence_outstanding,
                                    use_cluster=use_cluster)
                                pipeline_fence_wait(use_cluster=use_cluster)

                                _cur_b_addr_lo = _issue_active_b_tdm_only(
                                    _lb, _cur_b_addr_lo)

                                def _mid_issue_scalar(_mid_kb=_kb, _mid_lb=_lb):
                                    _issue_scalar_loads(_mid_kb, _mid_lb)

                                if const_expr(_use_scheduled_compute):
                                    rocdl.sched_barrier(0)
                                _acc = _compute_with_mid_loads(
                                    _acc,
                                    _bi,
                                    _mid_issue_scalar,
                                )
                                if const_expr(_use_scheduled_compute):
                                    _hot_loop_scheduler_scheduled()
                            _res = yield list(_acc) + [_cur_b_addr_lo]
                        acc = list(_res[:n_accs])
                        active_b_addr_lo = _res[n_accs]
                    else:
                        _init = list(acc)
                        for _li, _st in fx.range(0, loop_iters, 1, init=_init):
                            _acc = list(_st[:n_accs]) if isinstance(_st, (list, tuple)) else [_st]
                            for _bi in range_constexpr(_nb):
                                _lb = (_bi + _nb - 1) % _nb
                                _kt = (_li * fx.Index(_nb)
                                       + fx.Index(pre_loaded + _bi))
                                _kb = _kt * fx.Index(int(tile_k))
                                pipeline_fence_signal(
                                    outstanding=_fence_outstanding,
                                    use_cluster=use_cluster)
                                pipeline_fence_wait(use_cluster=use_cluster)

                                _issue_b_tdm_only(_kb, _lb)

                                def _mid_issue_scalar(_mid_kb=_kb, _mid_lb=_lb):
                                    _issue_scalar_loads(_mid_kb, _mid_lb)

                                if const_expr(_use_scheduled_compute):
                                    rocdl.sched_barrier(0)
                                _acc = _compute_with_mid_loads(
                                    _acc,
                                    _bi,
                                    _mid_issue_scalar,
                                )
                                if const_expr(_use_scheduled_compute):
                                    _hot_loop_scheduler_scheduled()
                            _res = yield list(_acc)
                        acc = list(_res[:n_accs]) if isinstance(_res, (list, tuple)) else [_res]

                # ── post-loop fence ──
                if const_expr(loop_iters > 0):
                    pipeline_fence(outstanding=0, use_cluster=use_cluster)
                elif const_expr(use_cluster):
                    gpu.cluster_barrier()

                # ── tail ──
                _tail_li = 0
                _tail_had_load = False
                for _ls, _cs, _out in _tail_plan:
                    if const_expr(_out == -1):
                        if const_expr(_tail_had_load):
                            pipeline_fence(outstanding=0,
                                           use_cluster=use_cluster)
                        if const_expr(_use_scheduled_compute):
                            rocdl.sched_barrier(0)
                            acc = _compute_k_tile_scheduled(acc, _cs)
                            _hot_loop_scheduler_scheduled()
                        else:
                            acc = _compute_k_tile(acc, _cs)
                    else:
                        pipeline_fence_signal(outstanding=_out,
                                              use_cluster=use_cluster)
                        pipeline_fence_wait(use_cluster=use_cluster)
                        if const_expr(_ls is not None):
                            _tail_had_load = True
                            _tkb = fx.Index(
                                (_tail_start + pre_loaded + _tail_li)
                                * int(tile_k))
                            _tail_li += 1

                            if const_expr(wave_specialized_tdm):
                                active_b_addr_lo = _issue_active_b_tdm_only(
                                    _ls, active_b_addr_lo)
                            else:
                                _issue_b_tdm_only(_tkb, _ls)

                            def _tail_mid_issue_scalar(_mid_kb=_tkb, _mid_ls=_ls):
                                _issue_scalar_loads(_mid_kb, _mid_ls)

                            if const_expr(_use_scheduled_compute):
                                rocdl.sched_barrier(0)
                            acc = _compute_with_mid_loads(
                                acc,
                                _cs,
                                _tail_mid_issue_scalar,
                            )
                            if const_expr(_use_scheduled_compute):
                                _hot_loop_scheduler_scheduled()
                        else:
                            if const_expr(_use_scheduled_compute):
                                rocdl.sched_barrier(0)
                                acc = _compute_k_tile_scheduled(acc, _cs)
                                _hot_loop_scheduler_scheduled()
                            else:
                                acc = _compute_k_tile(acc, _cs)

            out_elem_ty = _moe_out_elem_ty(out_dtype, T)

            if const_expr(bool(use_tdm_store)):
                # ── TDM store epilogue: acc → LDS → global (contiguous sorted output) ──
                # Pre-compute per-wm row scale (weight × validity mask)
                _scale_per_wm = []
                for _wm in range_constexpr(wmma_m_rep):
                    _m_off_val = _wm * WMMA_M
                    _row_local = warp_m_base + arith.index(_m_off_val) + lane16
                    _sorted_row = by * arith.index(int(tile_m)) + _row_local
                    _sorted_i32 = arith.index_cast(T.i32, _sorted_row)
                    _row_in_route = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        arith.index_cast(T.i32, _row_local),
                        arith.constant(int(route_tile_m), type=T.i32))
                    _row_in_valid = arith.cmpi(
                        arith.CmpIPredicate.slt, _sorted_i32, num_valid_i32)
                    _row_ok = arith.andi(_row_in_route, _row_in_valid)
                    if const_expr(bool(doweight_stage2)):
                        _sorted_safe = arith.select(
                            _row_ok, _sorted_i32, block_row_start)
                        _tw = buffer_ops.buffer_load(
                            tw_rsrc, _sorted_safe, vec_width=1, dtype=T.f32)
                        _sc = arith.select(
                            _row_ok, _tw,
                            arith.constant(0.0, type=T.f32))
                    else:
                        _sc = arith.select(
                            _row_ok,
                            arith.constant(1.0, type=T.f32),
                            arith.constant(0.0, type=T.f32))
                    _scale_per_wm.append(_sc)

                if const_expr(d_need_epilogue_fence):
                    pipeline_fence(outstanding=0, use_cluster=use_cluster)
                rocdl.sched_barrier(0)

                for _acc_idx, _vec_base, _m_off, _wn in _sub_tiles:
                    _wm_idx = _m_off // WMMA_M
                    _sc = _scale_per_wm[_wm_idx]
                    _sub8 = _extract_sub8(
                        acc[_acc_idx], _vec_base,
                        vector=vector,
                        range_constexpr=range_constexpr,
                        ACC_VEC_SIZE=ACC_VEC_SIZE)
                    _scaled = []
                    for _vi in range_constexpr(8):
                        _v = vector.extract(
                            _sub8,
                            static_position=[_vi],
                            dynamic_position=[])
                        _scaled.append(_v * _sc)
                    _scaled_sub8 = vector.from_elements(
                        T.vec(8, T.f32), _scaled)
                    _imm = _m_off * _lds_d_stride_elems + _wn * _n_col_d_elems
                    store_acc_vec8_to_lds(
                        d_lds_buffer, d_lane_base, _imm, _scaled_sub8,
                        out_elem=out_elem_ty)

                rocdl.s_wait_dscnt(0)
                tdm_ops.tensor_store_2d(d_desc)
                tdm_ops.tensor_wait(0)
            else:
                def _load_sub8(acc_idx, vec_base):
                    return _extract_sub8(
                        acc[acc_idx], vec_base, vector=vector, range_constexpr=range_constexpr, ACC_VEC_SIZE=ACC_VEC_SIZE
                    )

                _emit_stage2_store_epilogue(
                    sub_tiles=_sub_tiles,
                    by=by,
                    tile_m=int(tile_m),
                    route_tile_m=int(route_tile_m),
                    warp_m_base=warp_m_base,
                    warp_n_base=warp_n_base,
                    blk_n=blk_n,
                    lane16=lane16,
                    lane_kgrp=lane_kgrp,
                    WMMA_N=WMMA_N,
                    i32_tokens_in=i32_tokens_in,
                    i32_n_in=i32_n_in,
                    topk=int(topk),
                    num_valid_i32=num_valid_i32,
                    block_row_start=block_row_start,
                    lds_tid=lds_tid,
                    memref=memref,
                    sorted_rsrc=sorted_rsrc,
                    tw_rsrc=tw_rsrc,
                    out_rsrc=out_rsrc,
                    doweight_stage2=bool(doweight_stage2),
                    accumulate=bool(accumulate),
                    out_elem_ty=out_elem_ty,
                    load_sub8=_load_sub8,
                    ir=ir,
                    fx=fx,
                    arith=arith,
                    buffer_ops=buffer_ops,
                    scf=scf,
                    vector=vector,
                    range_constexpr=range_constexpr,
                    rocdl=rocdl,
                    T=T,
                )
            scf.YieldOp([])

    @flyc.jit
    def launch_mxscale_stage2_single(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_num_valid_ids: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_n_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
        stream: fx.Stream,
    ):
        _ = i32_k_in
        ctx = CompilationContext.get_current()
        n_in = arith.index_cast(T.index, i32_n_in)
        size_expert_ids_in = arith.index_cast(T.index, i32_size_expert_ids_in)
        gx = (n_in + fx.Index(int(tile_n) - 1)) // fx.Index(int(tile_n))
        gy = size_expert_ids_in
        launcher = moe_mxscale_stage2_single(
            arg_out, arg_x, arg_w, arg_scale_x, arg_scale_w,
            arg_sorted_token_ids, arg_expert_ids, arg_sorted_weights, arg_num_valid_ids,
            i32_tokens_in, i32_n_in, i32_k_in, i32_size_expert_ids_in,
        )
        _cluster_arg = (int(cluster_m), int(cluster_n), 1) if use_cluster else None
        _finalize_alloc_and_launch_2d(
            ctx=ctx,
            alloc=alloc,
            launcher=launcher,
            gx=gx,
            gy=gy,
            block_threads=block_threads,
            stream=stream,
            waves_per_eu=effective_waves_per_eu,
            ir=ir,
            cluster=_cluster_arg,
        )

    if expert_sched_mode:
        launch_mxscale_stage2_single.compile_hints["llvm_options"] = {
            "amdgpu-expert-scheduling-mode": True,
        }

    return launch_mxscale_stage2_single


# ---------------------------------------------------------------------------
# Public API entry points for fp4/fp8/a8w4
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1024)
def _compile_moe_mxscale_gemm(
    *,
    stage: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight: bool,
    in_dtype: str = "fp4",
    out_dtype: str = "f16",
    accumulate: bool = True,
    waves_per_eu: int | None = None,
    expert_sched_mode: bool = True,
    num_buffers: int = 1,
    use_tdm_gather: bool = True,
    use_tdm_gather_as: bool = True,
    use_tdm_store: bool = False,
    inst_prefetch: bool = False,
    wave_specialized_tdm: bool = False,
    cluster_m: int = 1,
    cluster_n: int = 1,
    k_batch: int = 1,
):
    _require_gfx1250()
    if waves_per_eu is not None and int(waves_per_eu) < 1:
        raise ValueError(f"waves_per_eu must be >= 1, got {waves_per_eu!r}")
    if in_dtype not in ("fp4", "fp8", "a8w4"):
        raise ValueError(
            f"Unsupported in_dtype for MXScale stage{stage}: {in_dtype!r}, "
            "expected 'fp4', 'fp8', or 'a8w4'"
        )

    single_tile_m, single_tile_n, single_m_warp, single_n_warp = _pick_mxscale_launch_shape(
        in_dtype, int(tile_m), int(tile_n),
    )
    common = dict(
        model_dim=int(model_dim), inter_dim=int(inter_dim),
        experts=int(experts), topk=int(topk),
        route_tile_m=int(tile_m),
        tile_m=int(single_tile_m), tile_n=int(single_tile_n), tile_k=int(tile_k),
        m_warp=int(single_m_warp), n_warp=int(single_n_warp),
        out_dtype=out_dtype, waves_per_eu=waves_per_eu, data_format=in_dtype,
        expert_sched_mode=expert_sched_mode, num_buffers=int(num_buffers),
        use_tdm_gather=bool(use_tdm_gather),
        use_tdm_gather_as=bool(use_tdm_gather_as),
        use_tdm_store=bool(use_tdm_store),
        inst_prefetch=bool(inst_prefetch), wave_specialized_tdm=bool(wave_specialized_tdm),
        cluster_m=int(cluster_m), cluster_n=int(cluster_n),
    )

    if stage == 1:
        exe = _compile_stage1_mxscale_kernel_impl(
            doweight_stage1=bool(doweight), k_batch=int(k_batch), **common)
        if (
            int(k_batch) == 1
            and in_dtype in ("fp8", "a8w4")
            and (int(inter_dim) % int(single_tile_n) == 0)
        ):
            return _Stage1GateUpPackedWrapper(
                exe,
                experts=int(experts), inter_dim=int(inter_dim),
                tile_n=int(single_tile_n),
                packed_cols_w=(int(model_dim) // 2) if in_dtype == "a8w4" else int(model_dim),
                packed_cols_scale=int(model_dim) // 32,
            )
        return exe

    if int(k_batch) != 1:
        raise ValueError(
            "split-K (k_batch>1) is only supported on stage1 for MXScale MoE")
    return _compile_stage2_mxscale_kernel_impl(
        doweight_stage2=bool(doweight), accumulate=bool(accumulate), **common,
    )


def compile_moe_gemm1(*, doweight_stage1, group_size=-1, use_cshuffle_epilog=None, k_batch=1, **kw):
    return _compile_moe_mxscale_gemm(
        stage=1, doweight=doweight_stage1, k_batch=int(k_batch), **kw)


def compile_moe_gemm2(*, doweight_stage2, accumulate=True, group_size=-1, use_cshuffle_epilog=None, **kw):
    return _compile_moe_mxscale_gemm(stage=2, doweight=doweight_stage2, accumulate=accumulate, **kw)


def compile_moe_gemm2_ex(*, mode=MoeGemm2Mode.ATOMIC, valid_mask=None, zero_intermediate=True, **kw):
    if mode == MoeGemm2Mode.REDUCE:
        gemm2_exe = compile_moe_gemm2(accumulate=False, **kw)
        out_s = str(kw.get("out_dtype", "f16")).strip().lower()
        if out_s in ("f16", "fp16", "half"):
            dtype_str = "f16"
        elif out_s in ("bf16", "bfloat16"):
            dtype_str = "bf16"
        else:
            dtype_str = "f32"
        reduce_exe = compile_moe_reduction(
            topk=kw["topk"], model_dim=kw["model_dim"],
            dtype_str=dtype_str, use_mask=(valid_mask is not None),
        )
        from kernels.moe_gemm_2stage import _MoeGemm2ReduceWrapper
        return _MoeGemm2ReduceWrapper(
            gemm2_exe=gemm2_exe, reduce_exe=reduce_exe,
            topk=kw["topk"], model_dim=kw["model_dim"],
            out_dtype_str=dtype_str,
            use_mask=(valid_mask is not None),
            zero_intermediate=zero_intermediate,
        )
    return compile_moe_gemm2(accumulate=True, **kw)
