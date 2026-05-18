"""Unified MXFP4/MXFP8/A8W4 GEMM kernel for gfx1250.

Supports FP4 (E2M1), FP8 (E4M3) and A8W4 (FP8 activation + FP4 weight)
data with E8M0 block scales via V_WMMA_SCALE instructions.
Select precision with ``data_format="fp4"|"fp8"|"a8w4"``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx

from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu, idx2crd, range_constexpr, rocdl, tdm_ops, vector
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr, check_smem_capacity
from aiter.ops.flydsl.kernels.gemm_common_gfx1250 import (
    extract_lds_base_idx, get_lds_memref,
    issue_tdm_loads,
    lds_load_b128_raw,
    pipeline_fence, pipeline_fence_signal, pipeline_fence_wait,
    store_acc_vec8_to_buffer, store_acc_vec8_to_lds,
)
from aiter.ops.flydsl.kernels.pipeline_utils import make_tail_plan, tdm_epilogue_fence_threshold_bytes

# Common constants
WMMA_M, WMMA_N, WMMA_K = 16, 16, 128
WAVE_SIZE = 32
SCALE_BLOCK = 32
SCALES_PER_WMMA = WMMA_K // SCALE_BLOCK  # 4

LDS_PAD_A_BYTES = 16
LDS_PAD_D_BYTES = 16


def compile_mxscale_gemm(
    *,
    data_format: str = "fp4",
    M: int = 0,
    N: int = 0,
    K: int,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    m_warp: int = 2,
    n_warp: int = 2,
    num_buffers: int = 2,
    waves_per_eu: int = None,
    l2_prefetch_distance: int = 2,
    cluster_m: int = 1,
    cluster_n: int = 1,
    use_tdm_store: bool = True,
    out_dtype: str = "f32",
    inst_prefetch: bool = False,
    wave_specialized_tdm: bool = False,
    split_k: int = 1,
    use_scale_opsel: bool = False,
    expert_sched_mode: bool = True,
    atomic_barrier_enable: bool = False,
    batch_count: int = 1,
    grouped_masked_m: bool = False,
    grouped_persistent_m: bool = False,
    persistent_workers: int | None = None,
):
    """Compile an MXFP4 or MXFP8 GEMM kernel with TDM async copy.

    Args:
        data_format: "fp4" for FP4/E2M1, "fp8" for FP8/E4M3.

    Data layout (both formats):
        A: [M, K_packed] uint8 (FP4: K_packed=K//2, FP8: K_packed=K)
        B: [N, K_packed] uint8, preshuffled (16x16 byte tiles)
        scale_A: [M, K//32] uint8 E8M0 (preshuffled)
        scale_B: [N, K//32] uint8 E8M0 (preshuffled)

    Returns a JitFunction:
        launch_fn(arg_c, arg_a, arg_b, arg_a_scale, arg_b_scale, M, N, stream)
    """
    if data_format not in ("fp4", "fp8", "a8w4"):
        raise ValueError(f"data_format must be 'fp4', 'fp8', or 'a8w4', got {data_format!r}")

    is_fp4 = data_format == "fp4"
    is_a8w4 = data_format == "a8w4"

    if out_dtype not in ("f32", "bf16", "f16"):
        raise ValueError(f"out_dtype must be 'f32', 'bf16', or 'f16', got {out_dtype!r}")
    elem_bytes_d = 2 if out_dtype in ("bf16", "f16") else 4

    if num_buffers not in (2, 3, 4):
        raise ValueError(f"num_buffers must be 2, 3, or 4, got {num_buffers}")
    if split_k < 1:
        raise ValueError(f"split_k must be >= 1, got {split_k}")
    if batch_count < 1:
        raise ValueError(f"batch_count must be >= 1, got {batch_count}")
    if grouped_masked_m and batch_count <= 1:
        raise ValueError("grouped_masked_m requires batch_count > 1")
    if grouped_persistent_m and not grouped_masked_m:
        raise ValueError("grouped_persistent_m requires grouped_masked_m=True")
    if grouped_persistent_m and (cluster_m > 1 or cluster_n > 1):
        raise ValueError("grouped_persistent_m currently requires cluster_m=cluster_n=1")
    if grouped_persistent_m:
        if persistent_workers is None:
            from aiter.jit.utils.chip_info import get_cu_num

            _persistent_workers = int(get_cu_num())
        else:
            _persistent_workers = int(persistent_workers)
        if _persistent_workers < 1:
            raise ValueError(f"persistent_workers must be >= 1, got {_persistent_workers}")
    else:
        _persistent_workers = 0

    use_cluster = cluster_m > 1 or cluster_n > 1
    if use_cluster:
        if cluster_m * cluster_n > 16:
            raise ValueError(
                f"cluster_m * cluster_n must be <= 16, got {cluster_m}*{cluster_n}")
    effective_waves_per_eu = waves_per_eu
    if use_cluster and effective_waves_per_eu is None:
        effective_waves_per_eu = 2

    num_warps = m_warp * n_warp
    block_threads = num_warps * WAVE_SIZE
    if block_threads > 1024:
        raise ValueError(
            f"block_threads must be <= 1024, got {block_threads}")

    if wave_specialized_tdm and num_warps != 4:
        raise ValueError(
            f"wave_specialized_tdm requires exactly 4 waves, got {num_warps}")

    # -- Format-dependent compile-time constants --
    # A8W4: activation is FP8 (PACK_FACTOR_A=1), weight is FP4 (PACK_FACTOR_B=2)
    if is_a8w4:
        PACK_FACTOR_A = 1   # FP8 activation
        PACK_FACTOR_B = 2   # FP4 weight
    elif is_fp4:
        PACK_FACTOR_A = 2
        PACK_FACTOR_B = 2
    else:
        PACK_FACTOR_A = 1
        PACK_FACTOR_B = 1

    WMMA_N_EFF = 32 if is_fp4 else 16   # N-cols covered per WMMA instruction
    ACC_VEC_SIZE = 16 if is_fp4 else 8   # accumulator vector width
    DS_LOADS_PER_A_FRAG = 2 if is_fp4 else 4

    packed_tile_k_a = tile_k // PACK_FACTOR_A
    packed_tile_k_b = tile_k // PACK_FACTOR_B
    scale_k_per_tile = tile_k // SCALE_BLOCK
    K_packed_a = K // PACK_FACTOR_A
    K_packed_b = K // PACK_FACTOR_B
    K_scale = K // SCALE_BLOCK
    split_k_chunk = K // split_k

    if K % tile_k != 0:
        raise ValueError(f"K must be divisible by tile_k={tile_k}, got K={K}")
    if K % split_k != 0:
        raise ValueError(f"K must be divisible by split_k={split_k}, got K={K}")
    if split_k_chunk % tile_k != 0:
        raise ValueError(
            f"K/split_k must be divisible by tile_k={tile_k}, got {split_k_chunk}")
    if tile_k % WMMA_K != 0:
        raise ValueError(f"tile_k must be a multiple of {WMMA_K}, got {tile_k}")
    if tile_m % WMMA_M != 0:
        raise ValueError(f"tile_m must be a multiple of {WMMA_M}, got {tile_m}")
    if tile_n % WMMA_N != 0:
        raise ValueError(f"tile_n must be a multiple of {WMMA_N}, got {tile_n}")
    if packed_tile_k_a % 4 != 0:
        raise ValueError(f"packed_tile_k_a must be a multiple of 4, got {packed_tile_k_a}")
    if packed_tile_k_b % 4 != 0:
        raise ValueError(f"packed_tile_k_b must be a multiple of 4, got {packed_tile_k_b}")
    if scale_k_per_tile % 4 != 0:
        raise ValueError(
            f"scale_k_per_tile must be a multiple of 4 (tile_k >= 128), got {scale_k_per_tile}")

    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp
    if warp_tile_m % WMMA_M != 0:
        raise ValueError(f"warp_tile_m={warp_tile_m} must be a multiple of {WMMA_M}")
    if warp_tile_n % WMMA_N_EFF != 0:
        raise ValueError(f"warp_tile_n={warp_tile_n} must be a multiple of {WMMA_N_EFF}")

    if split_k > 1 and use_tdm_store:
        raise ValueError("split_k > 1 currently requires use_tdm_store=False")

    num_k_tiles = split_k_chunk // tile_k
    if num_k_tiles < num_buffers:
        raise ValueError(
            f"{num_buffers}-stage buffering requires num_k_tiles >= {num_buffers}, "
            f"got {num_k_tiles}")

    gpu_arch = str(get_hip_arch())
    assert gpu_arch.startswith("gfx1250"), f"Expected gfx1250, got {gpu_arch}"

    k_wmma_steps = tile_k // WMMA_K

    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N_EFF
    n_accs = wmma_m_rep * wmma_n_rep
    # FP4 A/B swap: BScale rep derived from WMMA_M, not WMMA_N_EFF
    b_scale_load_rep = warp_tile_n // WMMA_M if is_fp4 else wmma_n_rep

    _b_frag_loads_per_wn = 2 if is_a8w4 else 4
    _bs_ds_loads = (
        wmma_n_rep * _b_frag_loads_per_wn
        + (b_scale_load_rep + 3) // 4
        + (wmma_m_rep + 3) // 4
    )

    lds_a_stride_bytes = packed_tile_k_a + LDS_PAD_A_BYTES

    lds_a_data_bytes = tile_m * lds_a_stride_bytes
    lds_b_data_bytes = tile_n * packed_tile_k_b
    _scale_guard_bytes = 16
    lds_a_scale_bytes = tile_m * scale_k_per_tile + _scale_guard_bytes
    lds_b_scale_bytes = tile_n * scale_k_per_tile + _scale_guard_bytes
    interleaved_scale_cols_a = wmma_m_rep * scale_k_per_tile
    interleaved_scale_cols_b = b_scale_load_rep * scale_k_per_tile

    def _align_up(value: int, align: int) -> int:
        if value % align == 0:
            return value
        return (value + align - 1) // align * align

    # TDM descriptors partition a tile cooperatively across ``num_warps`` by
    # deriving per-wave offsets from ``wave_id``. In wave-specialized mode we
    # dedicate one loader wave to each tensor (A/B/A_scale/B_scale), so each
    # active loader wave must issue a full-tile descriptor by itself.
    tdm_desc_num_warps = 1 if wave_specialized_tdm else num_warps

    # All pipeline stages share the same intra-stage layout. Keep that layout
    # unchanged and only remap each logical stage to a physical base inside one
    # LDS arena so TDM epilogue can alias the dead prefix of the arena.
    stage_layout = SmemAllocator(
        None, arch=gpu_arch, global_sym_name=f"mxscale_{data_format}_layout")
    stage_a_data_rel_off = stage_layout._align(stage_layout.ptr, 16)
    stage_layout.ptr = stage_a_data_rel_off + lds_a_data_bytes
    stage_b_data_rel_off = stage_layout._align(stage_layout.ptr, 16)
    stage_layout.ptr = stage_b_data_rel_off + lds_b_data_bytes
    stage_a_scale_rel_off = stage_layout._align(stage_layout.ptr, 16)
    stage_layout.ptr = stage_a_scale_rel_off + lds_a_scale_bytes
    stage_b_scale_rel_off = stage_layout._align(stage_layout.ptr, 16)
    stage_layout.ptr = stage_b_scale_rel_off + lds_b_scale_bytes
    stage_bytes = _align_up(stage_layout.ptr, 128)

    pre_loaded = num_buffers - 1
    loop_iters = (num_k_tiles - pre_loaded) // num_buffers
    _tail_start = loop_iters * num_buffers
    extra = num_k_tiles - _tail_start - pre_loaded
    _base_tail_plan = make_tail_plan(num_buffers, pre_loaded, extra)

    _last_compute_stage = _base_tail_plan[-1][1]

    stage_pitch_bytes = _align_up(stage_bytes, 1024)
    arena_alloc = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name=(
            f"mxscale_{data_format}_{tile_m}x{tile_n}x{tile_k}_"
            f"{m_warp}x{n_warp}_{num_buffers}buf_arena"),
    )

    stage_phys_order = [i for i in range(num_buffers) if i != _last_compute_stage]
    stage_phys_order.append(_last_compute_stage)
    stage_base_off = [0] * num_buffers
    for phys_i, logical_i in enumerate(stage_phys_order):
        stage_base_off[logical_i] = phys_i * stage_pitch_bytes
    arena_alloc.ptr = stage_pitch_bytes * num_buffers
    arena_total_bytes = arena_alloc.ptr
    epilogue_fence_threshold_bytes = tdm_epilogue_fence_threshold_bytes(
        stage_base_off=stage_base_off,
        tail_plan=_base_tail_plan,
        loop_iters=loop_iters,
        extra=extra,
    )

    stage_a_data_off = [
        stage_base_off[i] + stage_a_data_rel_off for i in range(num_buffers)
    ]
    stage_b_data_off = [
        stage_base_off[i] + stage_b_data_rel_off for i in range(num_buffers)
    ]
    stage_a_scale_off = [
        stage_base_off[i] + stage_a_scale_rel_off for i in range(num_buffers)
    ]
    stage_b_scale_off = [
        stage_base_off[i] + stage_b_scale_rel_off for i in range(num_buffers)
    ]

    if use_tdm_store:
        # TDM store copies the LDS tile as described; it does not de-pad rows.
        # Keep the output LDS tile tightly packed so the store extent is exactly
        # warp_tile_n columns. Padding here turns a 32-col warp store into a
        # 40-col store and can fault on the last N tile.
        lds_d_row_stride = warp_tile_n * elem_bytes_d
        warp_d_bytes = warp_tile_m * lds_d_row_stride
        total_d_bytes = num_warps * warp_d_bytes
        d_output_off = 0
        _lds_d_stride_elems = lds_d_row_stride // 2
        _warp_d_elems = warp_d_bytes // 2
        _n_col_d_elems = WMMA_N * elem_bytes_d // 2
        d_need_epilogue_fence = total_d_bytes > epilogue_fence_threshold_bytes
        if total_d_bytes > arena_total_bytes:
            arena_total_bytes = total_d_bytes
            arena_alloc.ptr = total_d_bytes
    check_smem_capacity(arena_total_bytes, gpu_arch)

    # TENSORcnt is tracked per-wave in hardware. The regular path issues four
    # tensor ops per wave per K-stage, while the wave-specialized path issues
    # only one tensor op from each dedicated loader wave.
    TDM_LOADS_PER_STEP = 1 if wave_specialized_tdm else 4
    tail_plan = [
        (ls, cs, o * TDM_LOADS_PER_STEP // 2 if o > 0 else o)
        for ls, cs, o in _base_tail_plan
    ]

    # Pre-compute epilogue sub-tile layout (unified for FP4 vec16 and FP8 vec8)
    _sub_tiles = []
    for _wm in range(wmma_m_rep):
        for _wn in range(wmma_n_rep):
            if is_fp4:
                # vec<16,f32>: split into 2 x 8 elements (2 x 16-col halves)
                for _half in range(2):
                    acc_idx = _wm * wmma_n_rep + _wn
                    vec_base = _half * 8
                    m_off = _wm * WMMA_M
                    n_sub = _wn * 2 + _half
                    _sub_tiles.append((acc_idx, vec_base, m_off, n_sub))
            else:
                # vec<8,f32>: single 8-element block
                acc_idx = _wm * wmma_n_rep + _wn
                m_off = _wm * WMMA_M
                n_sub = _wn
                _sub_tiles.append((acc_idx, 0, m_off, n_sub))

    COMPUTE_SCHEDULE_ROW_MAJOR_STREAMING = "row_major_streaming"
    COMPUTE_SCHEDULE_FP4_COL_BAND = "fp4_col_band"

    def _pick_compute_schedule_kind():
        # The FP4 col-band (quadrant) schedule reduces VGPR bank conflicts by
        # splitting B loads into left/right halves and processing four quadrants
        # (top-left, bottom-left, top-right, bottom-right).  This distributes
        # accumulator writes across different VGPR bank groups and overlaps
        # B-right loading with quadrant-1 WMMA compute.
        if not is_fp4:
            return COMPUTE_SCHEDULE_ROW_MAJOR_STREAMING
        if wmma_m_rep % 2 != 0 or wmma_n_rep % 2 != 0:
            return COMPUTE_SCHEDULE_ROW_MAJOR_STREAMING
        if n_accs < 8:
            return COMPUTE_SCHEDULE_ROW_MAJOR_STREAMING
        return COMPUTE_SCHEDULE_FP4_COL_BAND

    compute_schedule_kind = _pick_compute_schedule_kind()
    use_fp4_bank_friendly_schedule = (
        compute_schedule_kind == COMPUTE_SCHEDULE_FP4_COL_BAND
    )
    needs_grouped_row_masked_store = grouped_masked_m and (M % tile_m != 0)

    if use_fp4_bank_friendly_schedule:
        _bank_half_wm = wmma_m_rep // 2
        _bank_half_wn = wmma_n_rep // 2
        _bank_group_size = _bank_half_wm * _bank_half_wn
        _bank_half_b_scale_rep = b_scale_load_rep // 2
        _bank_group_to_row_major = []
        for _wm in range(_bank_half_wm):
            for _wn in range(_bank_half_wn):
                _bank_group_to_row_major.append(_wm * wmma_n_rep + _wn)
        for _wm in range(_bank_half_wm, wmma_m_rep):
            for _wn in range(_bank_half_wn):
                _bank_group_to_row_major.append(_wm * wmma_n_rep + _wn)
        for _wm in range(_bank_half_wm):
            for _wn in range(_bank_half_wn, wmma_n_rep):
                _bank_group_to_row_major.append(_wm * wmma_n_rep + _wn)
        for _wm in range(_bank_half_wm, wmma_m_rep):
            for _wn in range(_bank_half_wn, wmma_n_rep):
                _bank_group_to_row_major.append(_wm * wmma_n_rep + _wn)

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def kernel_mxscale_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_a_scale: fx.Tensor,
        arg_b_scale: fx.Tensor,
        arg_masked_m: fx.Tensor,
        arg_m_tile_prefix: fx.Tensor,
        arg_m_tile_map: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
    ):
        # Enable back-to-back WMMA issue (SCHED_MODE bit[4] = DISABLE_VALU_STALL)
        rocdl.disable_xdl_arb_stall()

        if const_expr(inst_prefetch):
            from flydsl._mlir.dialects import llvm as llvm_dialect
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

        tx = gpu.thread_id("x")
        bx = arith.index_cast(T.index, _raw(gpu.block_id("x")))
        by = arith.index_cast(T.index, _raw(gpu.block_id("y")))
        m_tiles_per_batch = (arith.index(M) + arith.index(tile_m - 1)) / arith.index(tile_m)

        def _emit_tile(batch_idx, bx_local, by_local, bz, tile_valid_override=None):
            blk_m = bx_local * arith.index(tile_m)
            blk_n = by_local * arith.index(tile_n)
            split_k_base = bz * arith.index(split_k_chunk)
            batch_m_base = batch_idx * arith.index(M)
            batch_b_base = batch_idx * arith.index(N // 16)
            batch_as_base = batch_idx * arith.index(M // wmma_m_rep)
            batch_bs_base = batch_idx * arith.index(N // b_scale_load_rep)
            tile_valid = arith.constant(1, type=ir.IntegerType.get_signless(1))
            valid_m_i32 = i32_m.ir_value()
            if const_expr(grouped_masked_m):
                masked_m_rsrc = buffer_ops.create_buffer_resource(arg_masked_m, max_size=True)
                valid_m_i32 = buffer_ops.buffer_load(
                    masked_m_rsrc,
                    arith.index_cast(T.i32, batch_idx),
                    vec_width=1,
                    dtype=T.i32,
                )
                if const_expr(grouped_persistent_m):
                    if tile_valid_override is not None:
                        tile_valid = tile_valid_override
                else:
                    tile_valid = arith.cmpi(
                        arith.CmpIPredicate.slt,
                        arith.index_cast(T.i32, blk_m),
                        valid_m_i32,
                    )

            if const_expr(use_cluster):
                local_x, local_y = gpu.compute_cluster_position()
                a_mcast_mask, b_mcast_mask = gpu.compute_mcast_masks(
                    local_x, local_y, cluster_m, cluster_n)
            else:
                a_mcast_mask = 0
                b_mcast_mask = 0

            layout_thr = fx.make_layout(
                (m_warp, n_warp, 2, 16),
                (n_warp * WAVE_SIZE, WAVE_SIZE, 16, 1))
            thr_coord = idx2crd(tx, layout_thr)
            wave_m_idx, wave_n_idx, lane_kgrp, lane16 = (
                fx.get(thr_coord, 0), fx.get(thr_coord, 1),
                fx.get(thr_coord, 2), fx.get(thr_coord, 3))

            warp_m_base = wave_m_idx * arith.index(warp_tile_m)
            warp_n_base = wave_n_idx * arith.index(warp_tile_n)

            m_idx = arith.index_cast(T.index, i32_m.ir_value())
            n_stride = arith.index(N)
            if const_expr(batch_count > 1):
                c_rows = arith.index(batch_count * M)
            else:
                c_rows = m_idx
            c_nrec = c_rows * n_stride * arith.index(elem_bytes_d)
            c_rsrc = buffer_ops.create_buffer_resource(arg_c, num_records_bytes=c_nrec)
            zero_i32 = arith.constant(0, type=T.i32)

            def make_desc_a(memref, k_base):
                k_packed_off = k_base / arith.index(PACK_FACTOR_A)
                return tdm_ops.make_tensor_descriptor_2d(
                    global_ptr=arg_a, lds_memref=memref,
                    global_offset=(batch_m_base + blk_m, k_packed_off),
                    tensor_shape=(batch_count * M, K_packed_a),
                    strides=(K_packed_a, 1),
                    tile_shape=(tile_m, packed_tile_k_a),
                    elem_bytes=1,
                    pad_interval=packed_tile_k_a, pad_amount=LDS_PAD_A_BYTES,
                    num_warps=tdm_desc_num_warps,
                    workgroup_mask=a_mcast_mask,
                    atomic_barrier_enable=atomic_barrier_enable)

            def make_desc_b(memref, k_base):
                k_packed_off = k_base / arith.index(PACK_FACTOR_B)
                return tdm_ops.make_tensor_descriptor_2d(
                    global_ptr=arg_b, lds_memref=memref,
                    global_offset=(batch_b_base + blk_n / arith.index(16),
                                   k_packed_off * arith.index(16)),
                    tensor_shape=(batch_count * (N // 16), K_packed_b * 16),
                    strides=(K_packed_b * 16, 1),
                    tile_shape=(tile_n // 16, packed_tile_k_b * 16),
                    elem_bytes=1,
                    pad_interval=0, pad_amount=0,
                    num_warps=tdm_desc_num_warps,
                    workgroup_mask=b_mcast_mask,
                    atomic_barrier_enable=atomic_barrier_enable)

            def make_desc_as(memref, k_base):
                k_scale_off = k_base / arith.index(SCALE_BLOCK)
                outer_off = blk_m / arith.index(wmma_m_rep)
                inner_off = k_scale_off * arith.index(wmma_m_rep)
                return tdm_ops.make_tensor_descriptor_2d(
                    global_ptr=arg_a_scale, lds_memref=memref,
                    global_offset=(batch_as_base + outer_off, inner_off),
                    tensor_shape=(batch_count * (M // wmma_m_rep), K_scale * wmma_m_rep),
                    strides=(wmma_m_rep * K_scale, 1),
                    tile_shape=(WMMA_M * m_warp, interleaved_scale_cols_a),
                    elem_bytes=1,
                    pad_interval=0, pad_amount=0,
                    num_warps=tdm_desc_num_warps,
                    workgroup_mask=a_mcast_mask,
                    atomic_barrier_enable=atomic_barrier_enable)

            def make_desc_bs(memref, k_base):
                k_scale_off = k_base / arith.index(SCALE_BLOCK)
                outer_off = blk_n / arith.index(b_scale_load_rep)
                inner_off = k_scale_off * arith.index(b_scale_load_rep)
                return tdm_ops.make_tensor_descriptor_2d(
                    global_ptr=arg_b_scale, lds_memref=memref,
                    global_offset=(batch_bs_base + outer_off, inner_off),
                    tensor_shape=(batch_count * (N // b_scale_load_rep), K_scale * b_scale_load_rep),
                    strides=(b_scale_load_rep * K_scale, 1),
                    tile_shape=(WMMA_M * n_warp, interleaved_scale_cols_b),
                    elem_bytes=1,
                    pad_interval=0, pad_amount=0,
                    num_warps=tdm_desc_num_warps,
                    workgroup_mask=b_mcast_mask,
                    atomic_barrier_enable=atomic_barrier_enable)

            if const_expr(wave_specialized_tdm):
                tdm_wave_id = rocdl.wave_id()
                tdm_wave_is_a = arith.cmpi(
                    arith.CmpIPredicate.eq, tdm_wave_id,
                    arith.constant(0, type=T.i32))
                tdm_wave_is_b = arith.cmpi(
                    arith.CmpIPredicate.eq, tdm_wave_id,
                    arith.constant(1, type=T.i32))
                tdm_wave_is_as = arith.cmpi(
                    arith.CmpIPredicate.eq, tdm_wave_id,
                    arith.constant(2, type=T.i32))

                def _select_wave_tdm_value(a_value, b_value, as_value, bs_value):
                    result = arith.select(tdm_wave_is_as, as_value, bs_value)
                    result = arith.select(tdm_wave_is_b, b_value, result)
                    return arith.select(tdm_wave_is_a, a_value, result)

            elem_ty_lds = T.f16

            def _precompute_a_lane_bases(lds_ptr):
                """Precompute per-wm A fragment lane base addresses (byte offsets)."""
                row_base = (warp_m_base + lane16) * arith.index(lds_a_stride_bytes)
                # K-dimension interleaving: kgrp0/kgrp1 read alternating 128-bit chunks
                # All formats: kgrp offset = 16 bytes (one ds_load_b128 width)
                k_half_off = lane_kgrp * arith.index(16)
                bases = []
                for wm in range_constexpr(wmma_m_rep):
                    base = row_base + arith.index(wm * WMMA_M * lds_a_stride_bytes) + k_half_off
                    bases.append(base)
                return lds_ptr, bases


            def load_a_frag(lds_buffer, a_lane_base, ks):
                """Load one A-fragment from LDS.

                FP4: vec<8xi32> via 2 x ds_load_b128 (32 bytes per lane).
                FP8/A8W4: vec<16xi32> via 4 x ds_load_b128 (64 bytes per lane).
                  Interleaved K layout:
                  kgrp0 reads bytes [0:15],[32:47],[64:79],[96:111] (stride=32)
                  kgrp1 reads bytes [16:31],[48:63],[80:95],[112:127] (stride=32)
                """
                k_byte_off = arith.index(ks * WMMA_K // PACK_FACTOR_A)
                byte_off = a_lane_base + k_byte_off
                v0 = lds_load_b128_raw(lds_buffer, byte_off)
                if const_expr(is_fp4):
                    # Interleaved stride=32: +0, +32
                    v1 = lds_load_b128_raw(lds_buffer, byte_off + arith.index(32))
                    return vector.shuffle(v0, v1, list(range(8)))
                else:
                    # Interleaved stride=32: +0, +32, +64, +96
                    v1 = lds_load_b128_raw(lds_buffer, byte_off + arith.index(32))
                    v2 = lds_load_b128_raw(lds_buffer, byte_off + arith.index(64))
                    v3 = lds_load_b128_raw(lds_buffer, byte_off + arith.index(96))
                    v01 = vector.shuffle(v0, v1, list(range(8)))
                    v23 = vector.shuffle(v2, v3, list(range(8)))
                    return vector.shuffle(v01, v23, list(range(16)))

            def _precompute_b_lane_bases(lds_ptr):
                """Precompute per-wn B fragment lane base addresses (byte offsets).

                FP4: 2 bases per wn (32-col WMMA = 2 N-groups of 16).
                FP8: 1 base per wn (16-col WMMA = 1 N-group).
                A8W4: 1 base per wn (16-col WMMA, FP4 packed weight).

                K-dimension interleaving for FP8/A8W4:
                  kgrp0 and kgrp1 read alternating 16x16 tiles (stride = 2 tiles).
                  kgrp offset = 1 tile = 256 bytes.
                """
                _ngroup_stride = packed_tile_k_b * 16
                _n_group_base = arith.index(warp_tile_n // 16) * wave_n_idx
                row_off = lane16 * arith.index(16)
                # All formats: interleaved -- kgrp offset = 1 tile = 256 bytes
                k_tile_off = lane_kgrp * arith.index(256)
                bases = []
                if const_expr(is_fp4):
                    for wn_half in range_constexpr(wmma_n_rep * 2):
                        ngroup_off = _n_group_base * arith.index(_ngroup_stride) \
                            + arith.index(wn_half * _ngroup_stride)
                        bases.append(ngroup_off + row_off + k_tile_off)
                else:
                    # FP8 and A8W4: 1 base per wn (16-col WMMA)
                    for wn in range_constexpr(wmma_n_rep):
                        ngroup_off = _n_group_base * arith.index(_ngroup_stride) \
                            + arith.index(wn * _ngroup_stride)
                        bases.append(ngroup_off + row_off + k_tile_off)
                return lds_ptr, bases

            def load_b_frag(lds_buffer, b_lane_bases, wn, ks):
                """Load one B-fragment from preshuffled LDS.

                FP4: 32x128 -> vec<16xi32> from 2 N-groups (bases[wn*2], bases[wn*2+1]).
                FP8: 16x128 -> vec<16xi32> from 1 N-group (bases[wn]).
                A8W4: 16x128 FP4 -> vec<8xi32> from 1 N-group (bases[wn]).

                K-dimension interleaving (FP8/A8W4):
                  Stride = 2 tiles = 512 bytes between loads.
                  kgrp0 reads tiles 0,2,4,6; kgrp1 reads tiles 1,3,5,7.
                """
                if const_expr(is_fp4):
                    # FP4: 2 N-groups per wn, 4 tiles per N-group
                    # Interleaved stride=512 (2 tiles): kgrp0->tiles 0,2; kgrp1->tiles 1,3
                    _num_tiles = WMMA_K // PACK_FACTOR_B // 16  # 4 tiles total per N-group
                    k_subtile_off = arith.index(ks * _num_tiles * 256)
                    base0 = b_lane_bases[wn * 2] + k_subtile_off
                    v0 = lds_load_b128_raw(lds_buffer, base0)
                    v1 = lds_load_b128_raw(lds_buffer, base0 + arith.index(512))
                    base1 = b_lane_bases[wn * 2 + 1] + k_subtile_off
                    v2 = lds_load_b128_raw(lds_buffer, base1)
                    v3 = lds_load_b128_raw(lds_buffer, base1 + arith.index(512))
                    v01 = vector.shuffle(v0, v1, list(range(8)))
                    v23 = vector.shuffle(v2, v3, list(range(8)))
                    return vector.shuffle(v01, v23, list(range(16)))
                elif const_expr(is_a8w4):
                    # A8W4: FP4 weight, 4 tiles per N-group
                    # Interleaved stride=512: kgrp0->tiles 0,2; kgrp1->tiles 1,3
                    _num_tiles = WMMA_K // PACK_FACTOR_B // 16  # 4 tiles total
                    k_subtile_off = arith.index(ks * _num_tiles * 256)
                    base0 = b_lane_bases[wn] + k_subtile_off
                    v0 = lds_load_b128_raw(lds_buffer, base0)
                    v1 = lds_load_b128_raw(lds_buffer, base0 + arith.index(512))
                    return vector.shuffle(v0, v1, list(range(8)))
                else:
                    # FP8: 8 tiles per N-group
                    # Interleaved stride=512: kgrp0->tiles 0,2,4,6; kgrp1->tiles 1,3,5,7
                    _num_tiles = WMMA_K // PACK_FACTOR_B // 16  # 8 tiles total
                    k_subtile_off = arith.index(ks * _num_tiles * 256)
                    base0 = b_lane_bases[wn] + k_subtile_off
                    v0 = lds_load_b128_raw(lds_buffer, base0)
                    v1 = lds_load_b128_raw(lds_buffer, base0 + arith.index(512))
                    v2 = lds_load_b128_raw(lds_buffer, base0 + arith.index(1024))
                    v3 = lds_load_b128_raw(lds_buffer, base0 + arith.index(1536))
                    v01 = vector.shuffle(v0, v1, list(range(8)))
                    v23 = vector.shuffle(v2, v3, list(range(8)))
                    return vector.shuffle(v01, v23, list(range(16)))

            def _precompute_scale_lane_bases(lds_ptr, warp_base, reps,
                                             interleaved_cols):
                """Precompute scale lane bases (byte offsets)."""
                warp_lds_row = warp_base / arith.index(reps) + lane16
                base = warp_lds_row * arith.index(interleaved_cols)
                if const_expr(is_fp4 or is_a8w4):
                    # FP4/A8W4: always add lane_kgrp offset (no opsel on BScale)
                    base = base + lane_kgrp * arith.index(SCALES_PER_WMMA)
                else:
                    # FP8: conditional on opsel
                    if const_expr(use_scale_opsel):
                        base = base + lane_kgrp * arith.index(SCALES_PER_WMMA)
                return lds_ptr, [base]

            def load_scale_b128(lds_buffer, scale_base, reps, ks=0):
                """Load all wmma_rep scales via ds_load_b128(s) for K-subtile *ks*."""
                ks_byte_off = ks * reps * SCALES_PER_WMMA
                eff_base = scale_base if ks_byte_off == 0 \
                    else scale_base + arith.index(ks_byte_off)
                num_loads = (reps + 3) // 4
                vecs = []
                for ld in range_constexpr(num_loads):
                    off = eff_base if ld == 0 else eff_base + arith.index(ld * 16)
                    vecs.append(lds_load_b128_raw(lds_buffer, off))
                results = []
                for i in range_constexpr(reps):
                    vi = vector.extract(vecs[i // 4],
                                        static_position=[i % 4], dynamic_position=[])
                    results.append(vi)
                return results

            def load_scale_slice_b128(lds_buffer, scale_base, full_reps,
                                      rep_start, rep_count, ks=0):
                """Load a contiguous slice of packed scale VGPRs for one K-subtile."""
                ks_byte_off = (ks * full_reps + rep_start) * SCALES_PER_WMMA
                eff_base = scale_base if ks_byte_off == 0 \
                    else scale_base + arith.index(ks_byte_off)
                num_loads = (rep_count + 3) // 4
                vecs = []
                for ld in range_constexpr(num_loads):
                    off = eff_base if ld == 0 else eff_base + arith.index(ld * 16)
                    vecs.append(lds_load_b128_raw(lds_buffer, off))
                results = []
                for i in range_constexpr(rep_count):
                    vi = vector.extract(vecs[i // 4],
                                        static_position=[i % 4], dynamic_position=[])
                    results.append(vi)
                return results

            def _load_b_and_scales(b_buf, b_bases, bs_buf, bs_bases,
                                   as_buf, as_bases, ks):
                """Load B frags + all scales for one K-subtile."""
                b_frags = [load_b_frag(b_buf, b_bases, wn, ks)
                           for wn in range_constexpr(wmma_n_rep)]
                b_scales_all = load_scale_b128(bs_buf, bs_bases[0],
                                               b_scale_load_rep, ks)
                a_scales_all = load_scale_b128(as_buf, as_bases[0], wmma_m_rep, ks)
                if const_expr(is_fp4):
                    # FP4 32x16: scaleAType=0 fixed (no op_sel on BScale)
                    b_scales = b_scales_all
                    if const_expr(use_scale_opsel):
                        a_scales = a_scales_all[::2]
                    else:
                        a_scales = a_scales_all
                else:
                    # FP8/A8W4 16x16: both scales support op_sel
                    if const_expr(use_scale_opsel):
                        b_scales = b_scales_all[::2]
                        a_scales = a_scales_all[::2]
                    else:
                        b_scales = b_scales_all
                        a_scales = a_scales_all
                return b_frags, b_scales, a_scales

            def _emit_wmma(accs, wm, wn, a_frag, b_frags, a_scales, b_scales):
                """Emit one WMMA instruction (format-specific)."""
                idx = wm * wmma_n_rep + wn
                if const_expr(use_scale_opsel):
                    a_scale_idx = wm // 2
                    a_opsel = wm % 2
                else:
                    a_scale_idx = wm
                    a_opsel = 0

                if const_expr(is_fp4):
                    # 32x16 WMMA with A/B swap: SRC0=B, SRC1=A
                    accs[idx] = rocdl.wmma_scale_f32_32x16x128_f4(
                        T.vec(16, T.f32),
                        b_frags[wn], a_frag, accs[idx],
                        b_scales[wn * 2], a_scales[a_scale_idx],
                        scaleAType=0,
                        scaleBType=a_opsel,
                    )
                else:
                    # 16x16x128 WMMA: A8W4 (fmtA=FP4) or FP8 (fmtA=FP8)
                    if const_expr(use_scale_opsel):
                        b_scale_idx = wn // 2
                        b_opsel = wn % 2
                    else:
                        b_scale_idx = wn
                        b_opsel = 0
                    accs[idx] = rocdl.wmma_scale_f32_16x16x128_f8f6f4(
                        T.vec(8, T.f32),
                        b_frags[wn], a_frag, accs[idx],
                        b_scales[b_scale_idx], a_scales[a_scale_idx],
                        fmtA=4 if is_a8w4 else 0, fmtB=0,
                        scaleAType=b_opsel, scaleBType=a_opsel,
                    )

            def _a_streaming_compute(accs, a_buf, a_bases, b_frags, b_scales,
                                     a_scales, ks, emit_filler=None,
                                     next_bs_info=None,
                                     mid_compute_callback=None):
                """Half-based A-streaming with zigzag wn ordering.

                When *next_bs_info* is provided, the next K-subtile's B+scale
                loads are issued BEFORE the s_wait_dscnt so they overlap with
                the current WMMA execution (partial drain pattern).
                """
                next_result = None
                _front_wm = (wmma_m_rep + 1) // 2
                _back_wm = wmma_m_rep - _front_wm

                def _emit_rows(start_wm, a_frags):
                    for frag_i in range_constexpr(len(a_frags)):
                        wm = start_wm + frag_i
                        is_last = (wm == wmma_m_rep - 1)
                        if const_expr(is_last and emit_filler is not None):
                            rocdl.sched_barrier(0)
                            emit_filler()
                        for wn_raw in range_constexpr(wmma_n_rep):
                            wn = (wmma_n_rep - 1 - wn_raw) if (wm % 2 == 1) else wn_raw
                            _emit_wmma(accs, wm, wn, a_frags[frag_i], b_frags,
                                       a_scales, b_scales)

                a_frags_front = [load_a_frag(a_buf, a_bases[wm], ks)
                                 for wm in range_constexpr(_front_wm)]

                _use_partial_drain = (
                    next_bs_info is not None
                    and _front_wm * wmma_n_rep >= 4
                )

                if const_expr(_use_partial_drain):
                    nb_buf, nb_bases, nbs_buf, nbs_bases, \
                        nas_buf, nas_bases, n_ks = next_bs_info
                    next_result = _load_b_and_scales(
                        nb_buf, nb_bases, nbs_buf, nbs_bases,
                        nas_buf, nas_bases, n_ks)
                    rocdl.s_wait_dscnt(_bs_ds_loads)
                else:
                    rocdl.s_wait_dscnt(0)

                _emit_rows(0, a_frags_front)

                if const_expr(mid_compute_callback is not None):
                    rocdl.sched_barrier(0)
                    mid_compute_callback()

                if const_expr(_back_wm > 0):
                    a_frags_back = [load_a_frag(a_buf, a_bases[_front_wm + h], ks)
                                    for h in range_constexpr(_back_wm)]
                    _back_drain = _bs_ds_loads if _use_partial_drain else 0
                    rocdl.s_wait_dscnt(_back_drain)
                    _emit_rows(_front_wm, a_frags_back)

                if const_expr(_use_partial_drain):
                    return accs, next_result
                if const_expr(next_bs_info is not None):
                    nb_buf, nb_bases, nbs_buf, nbs_bases, \
                        nas_buf, nas_bases, n_ks = next_bs_info
                    next_result = _load_b_and_scales(
                        nb_buf, nb_bases, nbs_buf, nbs_bases,
                        nas_buf, nas_bases, n_ks)
                    return accs, next_result
                return accs

            # -- Compute on one LDS buffer --
            def compute_tile(accs_in, lds_a, lds_b, lds_as, lds_bs,
                             emit_filler=None, mid_compute_callback=None):
                current_accs = list(accs_in)
                a_buf, a_bases = _precompute_a_lane_bases(lds_a)
                b_buf, b_bases = _precompute_b_lane_bases(lds_b)
                as_buf, as_bases = _precompute_scale_lane_bases(
                    lds_as, warp_m_base, wmma_m_rep, interleaved_scale_cols_a)
                bs_buf, bs_bases = _precompute_scale_lane_bases(
                    lds_bs, warp_n_base, b_scale_load_rep, interleaved_scale_cols_b)

                if const_expr(k_wmma_steps == 1):
                    b_frags, b_scales, a_scales = _load_b_and_scales(
                        b_buf, b_bases, bs_buf, bs_bases, as_buf, as_bases, 0)
                    current_accs = _a_streaming_compute(
                        current_accs, a_buf, a_bases, b_frags, b_scales,
                        a_scales, 0, emit_filler=emit_filler,
                        mid_compute_callback=mid_compute_callback)
                else:
                    prev_b, prev_bs, prev_as = _load_b_and_scales(
                        b_buf, b_bases, bs_buf, bs_bases, as_buf, as_bases, 0)
                    for ks in range_constexpr(k_wmma_steps - 1):
                        _mid_cb = mid_compute_callback if ks == 0 else None
                        current_accs, (prev_b, prev_bs, prev_as) = \
                            _a_streaming_compute(
                                current_accs, a_buf, a_bases, prev_b, prev_bs,
                                prev_as, ks,
                                next_bs_info=(b_buf, b_bases, bs_buf, bs_bases,
                                              as_buf, as_bases, ks + 1),
                                mid_compute_callback=_mid_cb)
                    current_accs = _a_streaming_compute(
                        current_accs, a_buf, a_bases, prev_b, prev_bs, prev_as,
                        k_wmma_steps - 1, emit_filler=emit_filler)
                return current_accs

            def compute_tile_fp4_bank_friendly(
                accs_in, lds_a, lds_b, lds_as, lds_bs, emit_filler=None,
                mid_compute_callback=None,
            ):
                current_accs = list(accs_in)
                a_buf, a_bases = _precompute_a_lane_bases(lds_a)
                b_buf, b_bases = _precompute_b_lane_bases(lds_b)
                as_buf, as_bases = _precompute_scale_lane_bases(
                    lds_as, warp_m_base, wmma_m_rep, interleaved_scale_cols_a)
                bs_buf, bs_bases = _precompute_scale_lane_bases(
                    lds_bs, warp_n_base, b_scale_load_rep, interleaved_scale_cols_b)
                _b_half_scale_loads = (_bank_half_b_scale_rep + 3) // 4

                def _fp4_get_a_scale_and_opsel(a_scales_all, wm_idx):
                    if const_expr(use_scale_opsel):
                        return a_scales_all[(wm_idx // 2) * 2], wm_idx % 2
                    return a_scales_all[wm_idx], 0

                def _load_a_group(wm_base, wm_count, ks):
                    return [
                        load_a_frag(a_buf, a_bases[wm_base + wm_local], ks)
                        for wm_local in range_constexpr(wm_count)
                    ]

                def _load_b_half(wn_base, ks):
                    return [
                        load_b_frag(b_buf, b_bases, wn_base + wn_local, ks)
                        for wn_local in range_constexpr(_bank_half_wn)
                    ]

                def _load_b_half_bundle(wn_base, rep_start, ks):
                    b_frags = _load_b_half(wn_base, ks)
                    b_scales = load_scale_slice_b128(
                        bs_buf, bs_bases[0],
                        b_scale_load_rep, rep_start,
                        _bank_half_b_scale_rep, ks)
                    return b_frags, b_scales

                def _emit_group_rows(group_base, wm_base, a_frags, b_frags, a_scales,
                                     b_scales, row_start, row_count,
                                     emit_filler_now=False):
                    if const_expr(emit_filler_now and emit_filler is not None):
                        rocdl.sched_barrier(0)
                        emit_filler()
                    for row_offset in range_constexpr(row_count):
                        wm_local = row_start + row_offset
                        a_frag = a_frags[wm_local]
                        global_wm = wm_base + wm_local
                        a_scale, a_opsel = _fp4_get_a_scale_and_opsel(
                            a_scales, global_wm)
                        row_base = group_base + wm_local * _bank_half_wn
                        for wn_local in range_constexpr(_bank_half_wn):
                            idx = row_base + wn_local
                            current_accs[idx] = rocdl.wmma_scale_f32_32x16x128_f4(
                                T.vec(16, T.f32),
                                b_frags[wn_local], a_frag, current_accs[idx],
                                b_scales[wn_local * 2], a_scale,
                                scaleAType=0,
                                scaleBType=a_opsel,
                            )

                def _emit_group(group_base, wm_base, a_frags, b_frags, a_scales,
                                b_scales, emit_filler_now=False):
                    _emit_group_rows(
                        group_base, wm_base,
                        a_frags, b_frags,
                        a_scales, b_scales,
                        0, _bank_half_wm,
                        emit_filler_now=emit_filler_now,
                    )

                b_left_frags, b_left_scales = _load_b_half_bundle(0, 0, 0)

                for ks in range_constexpr(k_wmma_steps):
                    is_last_ks = ks == k_wmma_steps - 1
                    a_scales_all = load_scale_b128(as_buf, as_bases[0],
                                                   wmma_m_rep, ks)

                    a_top_frags = _load_a_group(0, _bank_half_wm, ks)
                    a_bottom_frags = _load_a_group(_bank_half_wm, _bank_half_wm, ks)

                    # Wait for bottom-A loads; top-A stays in flight during Q1.
                    rocdl.s_wait_dscnt(_bank_half_wm * DS_LOADS_PER_A_FRAG)

                    _emit_group(
                        0, 0,
                        a_top_frags,
                        b_left_frags,
                        a_scales_all,
                        b_left_scales,
                    )

                    if const_expr(ks == 0 and mid_compute_callback is not None):
                        rocdl.sched_barrier(0)
                        mid_compute_callback()

                    b_right_frags, b_right_scales = _load_b_half_bundle(
                        _bank_half_wn, _bank_half_b_scale_rep, ks)

                    # Hold only the next B half outstanding while the second
                    # quadrant consumes the current left-half fragments.
                    rocdl.s_wait_dscnt(_bank_half_wn * 4 + _b_half_scale_loads)

                    _emit_group(
                        _bank_group_size, _bank_half_wm,
                        a_bottom_frags,
                        b_left_frags,
                        a_scales_all,
                        b_left_scales,
                    )

                    if const_expr(not is_last_ks):
                        next_left_frags, next_left_scales = _load_b_half_bundle(
                            0, 0, ks + 1)
                        # Older right-half loads must be ready before consuming
                        # them, while the next ks left-half preload can remain in
                        # flight under the final two quadrants.
                        rocdl.s_wait_dscnt(
                            _bank_half_wn * 4 + _b_half_scale_loads)
                    else:
                        rocdl.s_wait_dscnt(0)

                    _emit_group(
                        _bank_group_size * 2, 0,
                        a_top_frags,
                        b_right_frags,
                        a_scales_all,
                        b_right_scales,
                    )
                    _emit_group(
                        _bank_group_size * 3, _bank_half_wm,
                        a_bottom_frags,
                        b_right_frags,
                        a_scales_all,
                        b_right_scales,
                        emit_filler_now=is_last_ks,
                    )

                    if const_expr(not is_last_ks):
                        b_left_frags = next_left_frags
                        b_left_scales = next_left_scales

                return current_accs

            def hot_loop_scheduler():
                _half_wm = wmma_m_rep // 2
                _half_wmma = _half_wm * wmma_n_rep
                _b_loads_per_frag = 2 if is_a8w4 else 4

                for _ks in range_constexpr(k_wmma_steps):
                    if const_expr(_ks == 0):
                        rocdl.sched_dsrd(wmma_n_rep * _b_loads_per_frag + 2
                                         + _half_wm * DS_LOADS_PER_A_FRAG)
                    else:
                        rocdl.sched_dsrd(_half_wm * DS_LOADS_PER_A_FRAG)
                    rocdl.sched_mfma(_half_wmma)
                    rocdl.sched_dsrd(_half_wm * DS_LOADS_PER_A_FRAG)
                    rocdl.sched_mfma(_half_wmma)
                    if const_expr(_ks < k_wmma_steps - 1):
                        rocdl.sched_dsrd(wmma_n_rep * _b_loads_per_frag + 2)
                rocdl.sched_barrier(0)

            def hot_loop_scheduler_fp4_bank_friendly():
                _a_all_loads = wmma_m_rep * DS_LOADS_PER_A_FRAG
                _a_scale_loads = (wmma_m_rep + 3) // 4
                _b_half_loads = _bank_half_wn * 4
                _b_half_scale_loads = (_bank_half_b_scale_rep + 3) // 4
                _group_wmma = _bank_group_size
                _right_half_loads = _b_half_loads + _b_half_scale_loads

                for _ks in range_constexpr(k_wmma_steps):
                    if const_expr(_ks == 0):
                        rocdl.sched_dsrd(
                            _a_all_loads + _a_scale_loads
                            + _b_half_loads + _b_half_scale_loads)
                    else:
                        rocdl.sched_dsrd(_a_all_loads + _a_scale_loads)
                    rocdl.sched_mfma(_group_wmma)
                    rocdl.sched_dsrd(_right_half_loads)
                    rocdl.sched_mfma(_group_wmma)
                    if const_expr(_ks < k_wmma_steps - 1):
                        rocdl.sched_dsrd(_right_half_loads)
                    rocdl.sched_mfma(_group_wmma)
                    rocdl.sched_mfma(_group_wmma)
                rocdl.sched_barrier(0)

            def compute_tile_scheduled(accs_in, lds_a, lds_b, lds_as, lds_bs,
                                       emit_filler=None,
                                       mid_compute_callback=None):
                if const_expr(compute_schedule_kind == COMPUTE_SCHEDULE_FP4_COL_BAND):
                    return compute_tile_fp4_bank_friendly(
                        accs_in, lds_a, lds_b, lds_as, lds_bs,
                        emit_filler=emit_filler,
                        mid_compute_callback=mid_compute_callback)
                return compute_tile(
                    accs_in, lds_a, lds_b, lds_as, lds_bs,
                    emit_filler=emit_filler,
                    mid_compute_callback=mid_compute_callback)

            def hot_loop_scheduler_scheduled():
                if const_expr(compute_schedule_kind == COMPUTE_SCHEDULE_FP4_COL_BAND):
                    hot_loop_scheduler_fp4_bank_friendly()
                else:
                    hot_loop_scheduler()

            # -- Epilogue (unified via _sub_tiles) --
            def _get_acc_sub8(accs, acc_idx, vec_base):
                """Extract 8-element sub-vector from accumulator."""
                if const_expr(ACC_VEC_SIZE == 8):
                    return accs[acc_idx]
                indices = [vec_base + i for i in range_constexpr(8)]
                return vector.shuffle(accs[acc_idx], accs[acc_idx], indices)

            def epilogue_prepare_addrs():
                addrs = []
                _bf16_out = out_dtype in ("bf16", "f16")
                for acc_idx, vec_base, m_off, wn in _sub_tiles:
                    row = batch_m_base + blk_m + warp_m_base + arith.index(m_off) + lane16
                    col_base = (blk_n + warp_n_base + arith.index(wn * WMMA_N)
                                + lane_kgrp * arith.index(8))
                    if const_expr(_bf16_out):
                        c_off_bytes = (row * n_stride + col_base) \
                            * arith.index(elem_bytes_d)
                        addrs.append(c_off_bytes)
                    else:
                        for half in range_constexpr(2):
                            col = col_base + arith.index(half * 4)
                            c_off = row * n_stride + col
                            addrs.append(c_off)
                return addrs

            _bf16_out = out_dtype in ("bf16", "f16")
            _out_elem_local = T.bf16 if out_dtype == "bf16" else \
                (T.f16 if out_dtype == "f16" else None)

            def epilogue_stores(final_accs, addrs):
                addr_idx = 0
                for acc_idx, vec_base, m_off, wn in _sub_tiles:
                    sub8 = _get_acc_sub8(final_accs, acc_idx, vec_base)
                    if const_expr(needs_grouped_row_masked_store):
                        row_local = blk_m + warp_m_base + arith.index(m_off) + lane16
                        row_valid = arith.cmpi(
                            arith.CmpIPredicate.slt,
                            arith.index_cast(T.i32, row_local),
                            valid_m_i32,
                        )
                        store_valid = arith.andi(tile_valid, row_valid)
                        store_if = scf.IfOp(store_valid, results_=[], has_else=False)
                        with ir.InsertionPoint(store_if.then_block):
                            if const_expr(_bf16_out):
                                store_acc_vec8_to_buffer(
                                    sub8, c_rsrc, addrs[addr_idx],
                                    out_elem=_out_elem_local, offset_is_bytes=True)
                            else:
                                store_acc_vec8_to_buffer(
                                    sub8, c_rsrc, addrs[addr_idx:addr_idx + 2])
                            scf.YieldOp([])
                        addr_idx += 1 if _bf16_out else 2
                    else:
                        if const_expr(_bf16_out):
                            addr_idx += store_acc_vec8_to_buffer(
                                sub8, c_rsrc, addrs[addr_idx],
                                out_elem=_out_elem_local, offset_is_bytes=True)
                        else:
                            addr_idx += store_acc_vec8_to_buffer(
                                sub8, c_rsrc, addrs[addr_idx:addr_idx + 2])

            def epilogue_lds_stores(final_accs, d_buf, d_base):
                for acc_idx, vec_base, m_off, wn in _sub_tiles:
                    sub8 = _get_acc_sub8(final_accs, acc_idx, vec_base)
                    imm = m_off * _lds_d_stride_elems + wn * _n_col_d_elems
                    store_acc_vec8_to_lds(d_buf, d_base, imm, sub8,
                                          out_elem=_out_elem_local)

            def _atomic_add_acc_vec8_to_buffer(acc_vec8, addr):
                if const_expr(_bf16_out):
                    h_vec = arith.trunc_f(T.vec(8, _out_elem_local), acc_vec8)
                    pair_ty = T.vec(2, _out_elem_local)
                    for pair in range_constexpr(4):
                        e0 = vector.extract(
                            h_vec, static_position=[pair * 2], dynamic_position=[])
                        e1 = vector.extract(
                            h_vec, static_position=[pair * 2 + 1], dynamic_position=[])
                        pair_vec = vector.from_elements(pair_ty, [e0, e1])
                        byte_off = arith.index_cast(T.i32, addr + arith.index(pair * 4))
                        rocdl.raw_ptr_buffer_atomic_fadd(
                            pair_vec, c_rsrc, byte_off, zero_i32, zero_i32)
                    return 1

                for half in range_constexpr(2):
                    base_addr = addr[half] if isinstance(addr, (list, tuple)) else addr
                    for vi in range_constexpr(4):
                        val = vector.extract(
                            acc_vec8,
                            static_position=[half * 4 + vi],
                            dynamic_position=[],
                        )
                        byte_off = arith.index_cast(
                            T.i32, (base_addr + arith.index(vi)) * arith.index(4))
                        rocdl.raw_ptr_buffer_atomic_fadd(
                            val, c_rsrc, byte_off, zero_i32, zero_i32)
                return 2

            def epilogue_atomic_adds(final_accs, addrs):
                addr_idx = 0
                for acc_idx, vec_base, m_off, wn in _sub_tiles:
                    sub8 = _get_acc_sub8(final_accs, acc_idx, vec_base)
                    if const_expr(needs_grouped_row_masked_store):
                        row_local = blk_m + warp_m_base + arith.index(m_off) + lane16
                        row_valid = arith.cmpi(
                            arith.CmpIPredicate.slt,
                            arith.index_cast(T.i32, row_local),
                            valid_m_i32,
                        )
                        store_valid = arith.andi(tile_valid, row_valid)
                        store_if = scf.IfOp(store_valid, results_=[], has_else=False)
                        with ir.InsertionPoint(store_if.then_block):
                            if const_expr(_bf16_out):
                                _atomic_add_acc_vec8_to_buffer(sub8, addrs[addr_idx])
                            else:
                                _atomic_add_acc_vec8_to_buffer(
                                    sub8, addrs[addr_idx:addr_idx + 2])
                            scf.YieldOp([])
                        addr_idx += 1 if _bf16_out else 2
                    else:
                        if const_expr(_bf16_out):
                            addr_idx += _atomic_add_acc_vec8_to_buffer(
                                sub8, addrs[addr_idx])
                        else:
                            addr_idx += _atomic_add_acc_vec8_to_buffer(
                                sub8, addrs[addr_idx:addr_idx + 2])

            def grouped_accs_to_row_major(accs_grouped):
                row_major = [None] * n_accs
                for group_idx in range_constexpr(n_accs):
                    row_major[_bank_group_to_row_major[group_idx]] = accs_grouped[group_idx]
                return row_major

            def finalize_acc_layout(accs_in):
                if const_expr(compute_schedule_kind == COMPUTE_SCHEDULE_FP4_COL_BAND):
                    return grouped_accs_to_row_major(accs_in)
                return accs_in

            _effective_l2_pf = l2_prefetch_distance
            if const_expr(use_cluster and l2_prefetch_distance > 0):
                _effective_l2_pf = max(1, l2_prefetch_distance - 1)

            def _l2_prefetch(k_base):
                if const_expr(_effective_l2_pf <= 0):
                    return
                pf_k = k_base + arith.index(_effective_l2_pf * tile_k)
                pf_k_packed_a = pf_k / arith.index(PACK_FACTOR_A)
                pf_k_packed_b = pf_k / arith.index(PACK_FACTOR_B)
                tdm_ops.l2_prefetch_tile(
                    arg_a, (batch_m_base + blk_m, pf_k_packed_a),
                    (tile_m, packed_tile_k_a), (K_packed_a, 1),
                    elem_bytes=1, thread_id=tx, block_threads=block_threads)
                tdm_ops.l2_prefetch_tile(
                    arg_b,
                    (batch_b_base + blk_n / arith.index(16), pf_k_packed_b * arith.index(16)),
                    (tile_n // 16, packed_tile_k_b * 16),
                    (K_packed_b * 16, 1),
                    elem_bytes=1, thread_id=tx, block_threads=block_threads)

            # ====== Multi-stage pipeline ======
            acc_zero = arith.constant_vector(0.0, T.vec(ACC_VEC_SIZE, T.f32))
            accs = [acc_zero] * n_accs

            lds_a_data_f16 = lds_a_data_bytes // 2
            lds_b_data_f16 = lds_b_data_bytes // 2
            lds_a_scale_f16 = lds_a_scale_bytes // 2
            lds_b_scale_f16 = lds_b_scale_bytes // 2

            arena_base_ptr = arena_alloc.get_base()

            stages_a = [
                SmemPtr(arena_base_ptr, stage_a_data_off[i], elem_ty_lds,
                        shape=(lds_a_data_f16,))
                for i in range_constexpr(num_buffers)
            ]
            stages_b = [
                SmemPtr(arena_base_ptr, stage_b_data_off[i], elem_ty_lds,
                        shape=(lds_b_data_f16,))
                for i in range_constexpr(num_buffers)
            ]
            stages_as = [
                SmemPtr(arena_base_ptr, stage_a_scale_off[i], elem_ty_lds,
                        shape=(lds_a_scale_f16,))
                for i in range_constexpr(num_buffers)
            ]
            stages_bs = [
                SmemPtr(arena_base_ptr, stage_b_scale_off[i], elem_ty_lds,
                        shape=(lds_b_scale_f16,))
                for i in range_constexpr(num_buffers)
            ]

            stages_a_mem = [stages_a[i].get() for i in range_constexpr(num_buffers)]
            stages_b_mem = [stages_b[i].get() for i in range_constexpr(num_buffers)]
            stages_as_mem = [stages_as[i].get() for i in range_constexpr(num_buffers)]
            stages_bs_mem = [stages_bs[i].get() for i in range_constexpr(num_buffers)]

            stages_a_idx = [extract_lds_base_idx(stages_a[i])
                            for i in range_constexpr(num_buffers)]
            stages_b_idx = [extract_lds_base_idx(stages_b[i])
                            for i in range_constexpr(num_buffers)]
            stages_as_idx = [extract_lds_base_idx(stages_as[i])
                             for i in range_constexpr(num_buffers)]
            stages_bs_idx = [extract_lds_base_idx(stages_bs[i])
                             for i in range_constexpr(num_buffers)]

            if const_expr(use_tdm_store and not needs_grouped_row_masked_store):
                d_lds_base_ptr = arena_base_ptr
                d_lds_f16_count = total_d_bytes // 2
                d_smem = SmemPtr(d_lds_base_ptr, d_output_off, elem_ty_lds,
                                 shape=(d_lds_f16_count,))
                d_lds_buffer = get_lds_memref(d_smem)
                warp_lds_off = (wave_m_idx * arith.index(n_warp) + wave_n_idx) \
                    * arith.index(_warp_d_elems)
                d_lane_base = (warp_lds_off
                               + lane16 * arith.index(_lds_d_stride_elems)
                               + lane_kgrp * arith.index(4 * elem_bytes_d))
                wave_id_idx = arith.index_cast(T.index, rocdl.wave_id())
                d_warp_off_sgpr = wave_id_idx * arith.index(warp_d_bytes) \
                    + arith.index(d_output_off)
                warp_m_off_sgpr = (wave_id_idx / arith.index(n_warp)) \
                    * arith.index(warp_tile_m)
                warp_n_off_sgpr = (wave_id_idx % arith.index(n_warp)) \
                    * arith.index(warp_tile_n)
                d_desc = tdm_ops.make_tensor_descriptor_2d(
                    global_ptr=arg_c,
                    lds_memref=d_lds_base_ptr,
                    global_offset=(batch_m_base + blk_m + warp_m_off_sgpr,
                                   blk_n + warp_n_off_sgpr),
                    tensor_shape=(batch_count * M, N),
                    strides=(N, 1),
                    tile_shape=(warp_tile_m, warp_tile_n),
                    elem_bytes=elem_bytes_d,
                    pad_interval=0,
                    pad_amount=0,
                    num_warps=1,
                    lds_byte_offset=d_warp_off_sgpr,
                    for_store=True,
                )

            # Precompute LDS addresses for TDM descriptor switching
            stages_a_lds_addr = []
            stages_b_lds_addr = []
            stages_as_lds_addr = []
            stages_bs_lds_addr = []
            for i in range_constexpr(num_buffers):
                stages_a_lds_addr.append(vector.extract(
                    make_desc_a(stages_a_mem[i], arith.index(0)).dgroup0,
                    static_position=[1], dynamic_position=[]))
                stages_b_lds_addr.append(vector.extract(
                    make_desc_b(stages_b_mem[i], arith.index(0)).dgroup0,
                    static_position=[1], dynamic_position=[]))
                stages_as_lds_addr.append(vector.extract(
                    make_desc_as(stages_as_mem[i], arith.index(0)).dgroup0,
                    static_position=[1], dynamic_position=[]))
                stages_bs_lds_addr.append(vector.extract(
                    make_desc_bs(stages_bs_mem[i], arith.index(0)).dgroup0,
                    static_position=[1], dynamic_position=[]))

            desc_a_init = make_desc_a(stages_a_mem[0], split_k_base)
            desc_b_init = make_desc_b(stages_b_mem[0], split_k_base)
            desc_as_init = make_desc_as(stages_as_mem[0], split_k_base)
            desc_bs_init = make_desc_bs(stages_bs_mem[0], split_k_base)

            adv_a_i32 = arith.constant(tile_k // PACK_FACTOR_A, type=T.i32)
            adv_b_i32 = arith.constant(packed_tile_k_b * 16, type=T.i32)
            adv_as_i32 = arith.constant(tile_k // SCALE_BLOCK * wmma_m_rep, type=T.i32)
            adv_bs_i32 = arith.constant(tile_k // SCALE_BLOCK * b_scale_load_rep, type=T.i32)

            if const_expr(grouped_masked_m):
                pred_const = arith.select(
                    tile_valid,
                    arith.constant(1, type=T.i32),
                    arith.constant(0, type=T.i32),
                )
            else:
                pred_const = arith.constant(1, type=T.i32)

            if const_expr(wave_specialized_tdm):
                active_stage_lds_addr = [
                    _select_wave_tdm_value(
                        stages_a_lds_addr[i],
                        stages_b_lds_addr[i],
                        stages_as_lds_addr[i],
                        stages_bs_lds_addr[i],
                    )
                    for i in range_constexpr(num_buffers)
                ]
                active_addr_lo = _select_wave_tdm_value(
                    vector.extract(desc_a_init.dgroup0, static_position=[2], dynamic_position=[]),
                    vector.extract(desc_b_init.dgroup0, static_position=[2], dynamic_position=[]),
                    vector.extract(desc_as_init.dgroup0, static_position=[2], dynamic_position=[]),
                    vector.extract(desc_bs_init.dgroup0, static_position=[2], dynamic_position=[]),
                )
                active_addr_hi = _select_wave_tdm_value(
                    vector.extract(desc_a_init.dgroup0, static_position=[3], dynamic_position=[]),
                    vector.extract(desc_b_init.dgroup0, static_position=[3], dynamic_position=[]),
                    vector.extract(desc_as_init.dgroup0, static_position=[3], dynamic_position=[]),
                    vector.extract(desc_bs_init.dgroup0, static_position=[3], dynamic_position=[]),
                )
                active_dgroup1 = _select_wave_tdm_value(
                    desc_a_init.dgroup1,
                    desc_b_init.dgroup1,
                    desc_as_init.dgroup1,
                    desc_bs_init.dgroup1,
                )
                active_adv_i32 = _select_wave_tdm_value(
                    adv_a_i32, adv_b_i32, adv_as_i32, adv_bs_i32)
            else:
                addr_lo_a = vector.extract(desc_a_init.dgroup0, static_position=[2], dynamic_position=[])
                addr_hi_a = vector.extract(desc_a_init.dgroup0, static_position=[3], dynamic_position=[])
                addr_lo_b = vector.extract(desc_b_init.dgroup0, static_position=[2], dynamic_position=[])
                addr_hi_b = vector.extract(desc_b_init.dgroup0, static_position=[3], dynamic_position=[])
                addr_lo_as = vector.extract(desc_as_init.dgroup0, static_position=[2], dynamic_position=[])
                addr_hi_as = vector.extract(desc_as_init.dgroup0, static_position=[3], dynamic_position=[])
                addr_lo_bs = vector.extract(desc_bs_init.dgroup0, static_position=[2], dynamic_position=[])
                addr_hi_bs = vector.extract(desc_bs_init.dgroup0, static_position=[3], dynamic_position=[])

                dgroup1_a = desc_a_init.dgroup1
                dgroup1_b = desc_b_init.dgroup1
                dgroup1_as = desc_as_init.dgroup1
                dgroup1_bs = desc_bs_init.dgroup1

            # Prologue
            if const_expr(wave_specialized_tdm):
                for i in range_constexpr(pre_loaded):
                    dg0 = vector.from_elements(T.vec(4, T.i32), [
                        pred_const, active_stage_lds_addr[i],
                        active_addr_lo, active_addr_hi])
                    tdm_ops.tensor_load_2d(
                        tdm_ops.TDMDescriptor2D(dg0, active_dgroup1))
                    active_addr_lo = arith.addi(active_addr_lo, active_adv_i32)
            else:
                for i in range_constexpr(pre_loaded):
                    dg0_a = vector.from_elements(T.vec(4, T.i32), [
                        pred_const, stages_a_lds_addr[i], addr_lo_a, addr_hi_a])
                    dg0_b = vector.from_elements(T.vec(4, T.i32), [
                        pred_const, stages_b_lds_addr[i], addr_lo_b, addr_hi_b])
                    dg0_as = vector.from_elements(T.vec(4, T.i32), [
                        pred_const, stages_as_lds_addr[i], addr_lo_as, addr_hi_as])
                    dg0_bs = vector.from_elements(T.vec(4, T.i32), [
                        pred_const, stages_bs_lds_addr[i], addr_lo_bs, addr_hi_bs])

                    issue_tdm_loads(
                        tdm_ops.TDMDescriptor2D(dg0_a, dgroup1_a),
                        tdm_ops.TDMDescriptor2D(dg0_b, dgroup1_b),
                        tdm_ops.TDMDescriptor2D(dg0_as, dgroup1_as),
                        tdm_ops.TDMDescriptor2D(dg0_bs, dgroup1_bs),
                        wave_specialized=wave_specialized_tdm)

                    addr_lo_a = arith.addi(addr_lo_a, adv_a_i32)
                    addr_lo_b = arith.addi(addr_lo_b, adv_b_i32)
                    addr_lo_as = arith.addi(addr_lo_as, adv_as_i32)
                    addr_lo_bs = arith.addi(addr_lo_bs, adv_bs_i32)

            pipeline_fence(outstanding=TDM_LOADS_PER_STEP * (num_buffers - 2),
                           use_cluster=use_cluster)

            # Main loop -- acc_mixed style: fence at top, TDM_load mid-compute.
            # This overlaps TDM DMA with the remaining WMMA instructions,
            _fence_outstanding = TDM_LOADS_PER_STEP * (num_buffers - 2)

            if const_expr(loop_iters > 0):
                if const_expr(wave_specialized_tdm):
                    init_args = list(accs) + [active_addr_lo]

                    for loop_iter, state in range(0, loop_iters, 1, init=init_args):
                        accs_in = list(state[:n_accs])
                        cur_addr_lo = state[n_accs]

                        for buf_idx in range_constexpr(num_buffers):
                            load_stage = (buf_idx + num_buffers - 1) % num_buffers

                            pipeline_fence_signal(
                                outstanding=_fence_outstanding,
                                use_cluster=use_cluster)
                            pipeline_fence_wait(use_cluster=use_cluster)

                            addr_box = [cur_addr_lo]

                            def _mid_tdm_ws(
                                _ls=load_stage,
                                _ab=addr_box,
                                _k_off=(split_k_base
                                        + loop_iter * arith.index(num_buffers * tile_k)
                                        + arith.index(buf_idx * tile_k)),
                            ):
                                dg0 = vector.from_elements(T.vec(4, T.i32), [
                                    pred_const, active_stage_lds_addr[_ls],
                                    _ab[0], active_addr_hi])
                                tdm_ops.tensor_load_2d(
                                    tdm_ops.TDMDescriptor2D(dg0, active_dgroup1))
                                _ab[0] = arith.addi(_ab[0], active_adv_i32)
                                _l2_prefetch(_k_off)

                            rocdl.sched_barrier(0)
                            accs_in = compute_tile_scheduled(
                                accs_in,
                                stages_a_idx[buf_idx],
                                stages_b_idx[buf_idx],
                                stages_as_idx[buf_idx],
                                stages_bs_idx[buf_idx],
                                mid_compute_callback=_mid_tdm_ws)
                            cur_addr_lo = addr_box[0]
                            hot_loop_scheduler_scheduled()

                        results = yield list(accs_in) + [cur_addr_lo]

                    accs = list(results[:n_accs])
                    active_addr_lo = results[n_accs]
                else:
                    init_args = list(accs) + [addr_lo_a, addr_lo_b, addr_lo_as, addr_lo_bs]

                    for loop_iter, state in range(0, loop_iters, 1, init=init_args):
                        accs_in = list(state[:n_accs])
                        cur_lo_a = state[n_accs]
                        cur_lo_b = state[n_accs + 1]
                        cur_lo_as = state[n_accs + 2]
                        cur_lo_bs = state[n_accs + 3]

                        for buf_idx in range_constexpr(num_buffers):
                            load_stage = (buf_idx + num_buffers - 1) % num_buffers

                            pipeline_fence_signal(
                                outstanding=_fence_outstanding,
                                use_cluster=use_cluster)
                            pipeline_fence_wait(use_cluster=use_cluster)

                            addr_boxes = [[cur_lo_a], [cur_lo_b],
                                          [cur_lo_as], [cur_lo_bs]]

                            def _mid_tdm_nws(
                                _ls=load_stage,
                                _ab=addr_boxes,
                                _k_off=(split_k_base
                                        + loop_iter * arith.index(num_buffers * tile_k)
                                        + arith.index(buf_idx * tile_k)),
                            ):
                                dg0_a = vector.from_elements(T.vec(4, T.i32), [
                                    pred_const, stages_a_lds_addr[_ls],
                                    _ab[0][0], addr_hi_a])
                                dg0_b = vector.from_elements(T.vec(4, T.i32), [
                                    pred_const, stages_b_lds_addr[_ls],
                                    _ab[1][0], addr_hi_b])
                                dg0_as = vector.from_elements(T.vec(4, T.i32), [
                                    pred_const, stages_as_lds_addr[_ls],
                                    _ab[2][0], addr_hi_as])
                                dg0_bs = vector.from_elements(T.vec(4, T.i32), [
                                    pred_const, stages_bs_lds_addr[_ls],
                                    _ab[3][0], addr_hi_bs])
                                issue_tdm_loads(
                                    tdm_ops.TDMDescriptor2D(dg0_a, dgroup1_a),
                                    tdm_ops.TDMDescriptor2D(dg0_b, dgroup1_b),
                                    tdm_ops.TDMDescriptor2D(dg0_as, dgroup1_as),
                                    tdm_ops.TDMDescriptor2D(dg0_bs, dgroup1_bs),
                                    wave_specialized=wave_specialized_tdm)
                                _ab[0][0] = arith.addi(_ab[0][0], adv_a_i32)
                                _ab[1][0] = arith.addi(_ab[1][0], adv_b_i32)
                                _ab[2][0] = arith.addi(_ab[2][0], adv_as_i32)
                                _ab[3][0] = arith.addi(_ab[3][0], adv_bs_i32)
                                _l2_prefetch(_k_off)

                            rocdl.sched_barrier(0)
                            accs_in = compute_tile_scheduled(
                                accs_in,
                                stages_a_idx[buf_idx],
                                stages_b_idx[buf_idx],
                                stages_as_idx[buf_idx],
                                stages_bs_idx[buf_idx],
                                mid_compute_callback=_mid_tdm_nws)
                            cur_lo_a = addr_boxes[0][0]
                            cur_lo_b = addr_boxes[1][0]
                            cur_lo_as = addr_boxes[2][0]
                            cur_lo_bs = addr_boxes[3][0]
                            hot_loop_scheduler_scheduled()

                        results = yield list(accs_in) + \
                            [cur_lo_a, cur_lo_b, cur_lo_as, cur_lo_bs]

                    accs = list(results[:n_accs])
                    addr_lo_a = results[n_accs]
                    addr_lo_b = results[n_accs + 1]
                    addr_lo_as = results[n_accs + 2]
                    addr_lo_bs = results[n_accs + 3]

            # Tail -- same acc_mixed pattern: fence at top, TDM mid-compute.
            if const_expr(loop_iters > 0):
                pipeline_fence(outstanding=0, use_cluster=use_cluster)
            elif const_expr(use_cluster):
                gpu.cluster_barrier()
            epi_addrs_box = [None]
            _tail_had_load = False
            for _load_stage, _compute_stage, _outstanding in tail_plan:
                if const_expr(_outstanding == -1):
                    if const_expr(_tail_had_load):
                        pipeline_fence(outstanding=0, use_cluster=use_cluster)
                    if const_expr(use_tdm_store):
                        accs = compute_tile_scheduled(
                            accs,
                            stages_a_idx[_compute_stage], stages_b_idx[_compute_stage],
                            stages_as_idx[_compute_stage], stages_bs_idx[_compute_stage])
                    else:
                        def _emit_epi_addrs():
                            epi_addrs_box[0] = epilogue_prepare_addrs()

                        accs = compute_tile_scheduled(
                            accs,
                            stages_a_idx[_compute_stage], stages_b_idx[_compute_stage],
                            stages_as_idx[_compute_stage], stages_bs_idx[_compute_stage],
                            emit_filler=_emit_epi_addrs)
                else:
                    pipeline_fence_signal(outstanding=_outstanding,
                                          use_cluster=use_cluster)
                    pipeline_fence_wait(use_cluster=use_cluster)

                    _tail_mid_cb = None
                    if const_expr(_load_stage is not None):
                        _tail_had_load = True
                        if const_expr(wave_specialized_tdm):
                            _tail_addr_box = [active_addr_lo]

                            def _tail_mid_ws(_ls=_load_stage, _ab=_tail_addr_box):
                                dg0 = vector.from_elements(T.vec(4, T.i32), [
                                    pred_const, active_stage_lds_addr[_ls],
                                    _ab[0], active_addr_hi])
                                tdm_ops.tensor_load_2d(
                                    tdm_ops.TDMDescriptor2D(dg0, active_dgroup1))
                                _ab[0] = arith.addi(_ab[0], active_adv_i32)

                            _tail_mid_cb = _tail_mid_ws
                        else:
                            _tail_ab = [[addr_lo_a], [addr_lo_b],
                                        [addr_lo_as], [addr_lo_bs]]

                            def _tail_mid_nws(_ls=_load_stage, _ab=_tail_ab):
                                dg0_a = vector.from_elements(T.vec(4, T.i32), [
                                    pred_const, stages_a_lds_addr[_ls],
                                    _ab[0][0], addr_hi_a])
                                dg0_b = vector.from_elements(T.vec(4, T.i32), [
                                    pred_const, stages_b_lds_addr[_ls],
                                    _ab[1][0], addr_hi_b])
                                dg0_as = vector.from_elements(T.vec(4, T.i32), [
                                    pred_const, stages_as_lds_addr[_ls],
                                    _ab[2][0], addr_hi_as])
                                dg0_bs = vector.from_elements(T.vec(4, T.i32), [
                                    pred_const, stages_bs_lds_addr[_ls],
                                    _ab[3][0], addr_hi_bs])
                                issue_tdm_loads(
                                    tdm_ops.TDMDescriptor2D(dg0_a, dgroup1_a),
                                    tdm_ops.TDMDescriptor2D(dg0_b, dgroup1_b),
                                    tdm_ops.TDMDescriptor2D(dg0_as, dgroup1_as),
                                    tdm_ops.TDMDescriptor2D(dg0_bs, dgroup1_bs),
                                    wave_specialized=wave_specialized_tdm)
                                _ab[0][0] = arith.addi(_ab[0][0], adv_a_i32)
                                _ab[1][0] = arith.addi(_ab[1][0], adv_b_i32)
                                _ab[2][0] = arith.addi(_ab[2][0], adv_as_i32)
                                _ab[3][0] = arith.addi(_ab[3][0], adv_bs_i32)

                            _tail_mid_cb = _tail_mid_nws

                    rocdl.sched_barrier(0)
                    accs = compute_tile_scheduled(
                        accs,
                        stages_a_idx[_compute_stage], stages_b_idx[_compute_stage],
                        stages_as_idx[_compute_stage], stages_bs_idx[_compute_stage],
                        mid_compute_callback=_tail_mid_cb)

                    if const_expr(_load_stage is not None):
                        if const_expr(wave_specialized_tdm):
                            active_addr_lo = _tail_addr_box[0]
                        else:
                            addr_lo_a = _tail_ab[0][0]
                            addr_lo_b = _tail_ab[1][0]
                            addr_lo_as = _tail_ab[2][0]
                            addr_lo_bs = _tail_ab[3][0]

                    hot_loop_scheduler_scheduled()

            accs = finalize_acc_layout(accs)

            if const_expr(use_tdm_store and not needs_grouped_row_masked_store):
                if const_expr(d_need_epilogue_fence):
                    pipeline_fence(outstanding=0, use_cluster=use_cluster)
                rocdl.sched_barrier(0)
                epilogue_lds_stores(accs, d_lds_buffer, d_lane_base)
                rocdl.s_wait_dscnt(0)
                tdm_ops.tensor_store_2d(d_desc)
                tdm_ops.tensor_wait(0)
            else:
                rocdl.sched_barrier(0)
                if const_expr(epi_addrs_box[0] is None):
                    epi_addrs_box[0] = epilogue_prepare_addrs()
                if const_expr(split_k > 1):
                    epilogue_atomic_adds(accs, epi_addrs_box[0])
                else:
                    epilogue_stores(accs, epi_addrs_box[0])


        if const_expr(grouped_persistent_m):
            prefix_rsrc = buffer_ops.create_buffer_resource(arg_m_tile_prefix, max_size=True)
            map_rsrc = buffer_ops.create_buffer_resource(arg_m_tile_map, max_size=True)
            block_n_id = arith.index_cast(T.index, _raw(gpu.block_idx.x))
            worker_id = arith.index_cast(T.index, _raw(gpu.block_idx.y))
            grid_size = arith.index(_persistent_workers)
            idx_n = arith.index_cast(T.index, i32_n.ir_value())
            n_tiles_per_batch = (idx_n + arith.index(tile_n - 1)) / arith.index(tile_n)
            max_m_tiles_per_batch = (M + tile_m - 1) // tile_m

            # Total M tile count = prefix[batch_count].
            # Do not wrap _emit_tile in a dynamic Python `if`: in FlyDSL this is
            # easy to miscompile into a path that never emits the tile body.
            total_m_tiles_i32 = buffer_ops.buffer_load(
                prefix_rsrc, batch_count, vec_width=1, dtype=T.i32)
            total_m_tiles_idx = arith.index_cast(T.index, total_m_tiles_i32)
            tiles_per_worker = (
                total_m_tiles_idx + grid_size - arith.index(1)
            ) / grid_size

            # gfx950 a4w4 stage2 persist_m<=0 style: grid.x enumerates N tiles,
            # grid.y enumerates CU workers. Each worker owns a contiguous chunk
            # of M tiles for the fixed N tile, improving B reuse and avoiding a
            # global flattened tile stream inside the worker loop.
            _c0_p = arith.index(0)
            _c1_p = arith.index(1)
            _i1 = ir.IntegerType.get_signless(1)
            _init_active = arith.constant(1, type=_i1)
            _for_persist = scf.ForOp(_c0_p, tiles_per_worker, _c1_p, [_init_active])
            _for_ip = ir.InsertionPoint(_for_persist.body)
            _for_ip.__enter__()

            mi = _for_persist.induction_variable
            still_active = _for_persist.inner_iter_args[0]
            global_m_tile = worker_id * tiles_per_worker + mi
            m_tile_in_range = arith.cmpi(
                arith.CmpIPredicate.slt, global_m_tile, total_m_tiles_idx)
            n_tile_in_range = arith.cmpi(
                arith.CmpIPredicate.slt, block_n_id, n_tiles_per_batch)
            cur_active = arith.andi(still_active, m_tile_in_range)
            tile_active = arith.andi(cur_active, n_tile_in_range)

            # tile_active is uniform across the workgroup. Skipping the full
            # tile body for inactive persistent workers avoids burning WMMA
            # cycles on the common decode case where total_m_tiles << CU.
            tile_if = scf.IfOp(tile_active, results_=[], has_else=False)
            with ir.InsertionPoint(tile_if.then_block):
                map_entry_i32 = buffer_ops.buffer_load(
                    map_rsrc, global_m_tile, vec_width=1, dtype=T.i32)
                map_entry = arith.index_cast(T.index, map_entry_i32)
                batch_idx = map_entry / arith.index(max_m_tiles_per_batch)
                bx_local = map_entry - batch_idx * arith.index(max_m_tiles_per_batch)
                split_k_id = arith.index_cast(T.index, _raw(gpu.block_idx.z))
                _emit_tile(batch_idx, bx_local, block_n_id, split_k_id,
                           tile_valid_override=tile_active)
                scf.YieldOp([])
            gpu.barrier()
            scf.YieldOp([cur_active])
            _for_ip.__exit__(None, None, None)
        else:
            if const_expr(batch_count > 1):
                flat_bx = arith.index_cast(T.index, _raw(gpu.block_idx.x))
                batch_idx = flat_bx / m_tiles_per_batch
                bx_local = flat_bx - batch_idx * m_tiles_per_batch
                bz = arith.index_cast(T.index, _raw(gpu.block_idx.z)) if split_k > 1 else arith.index(0)
            else:
                batch_idx = arith.index(0)
                bx_local = bx
                bz = arith.index_cast(T.index, _raw(gpu.block_idx.z)) if split_k > 1 else arith.index(0)
            _emit_tile(batch_idx, bx_local, by, bz)

    # Bump this when changing generated IR in ways not otherwise reflected in
    # the shape/config tuple below. This forces FlyDSL's JIT/cache path to stop
    # reusing a previously compiled kernel after source-only descriptor fixes.
    tdm_store_descriptor_version = 18

    cache_tag = (data_format, K, tile_m, tile_n, tile_k, m_warp, n_warp,
                 num_buffers, compute_schedule_kind,
                 effective_waves_per_eu, l2_prefetch_distance,
                 cluster_m, cluster_n, use_tdm_store,
                 out_dtype, inst_prefetch, wave_specialized_tdm, split_k,
                 use_scale_opsel, expert_sched_mode,
                 atomic_barrier_enable, batch_count, grouped_masked_m,
                 grouped_persistent_m, _persistent_workers,
                 tdm_store_descriptor_version)

    @flyc.jit
    def launch_mxscale_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_a_scale: fx.Tensor,
        arg_b_scale: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        stream: fx.Stream,
    ):
        _ = cache_tag
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            arena_alloc.finalized = False
            arena_alloc.finalize()

        idx_m = arith.index_cast(T.index, i32_m.ir_value())
        idx_n = arith.index_cast(T.index, i32_n.ir_value())
        gx = _raw((idx_m + arith.index(tile_m - 1)) / arith.index(tile_m))
        if const_expr(batch_count > 1):
            gx = gx * batch_count
        gy = _raw((idx_n + arith.index(tile_n - 1)) / arith.index(tile_n))
        gz = split_k

        launcher = kernel_mxscale_gemm(
            arg_c, arg_a, arg_b, arg_a_scale, arg_b_scale,
            arg_c, arg_c, arg_c, i32_m, i32_n)
        for op in ctx.gpu_module_body.operations:
            if const_expr(hasattr(op, 'attributes') and op.OPERATION_NAME == "gpu.func"):
                if const_expr(effective_waves_per_eu is not None):
                    _wpe = int(effective_waves_per_eu)
                    if const_expr(_wpe >= 1):
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                            ir.IntegerType.get_signless(32), _wpe)
                if const_expr(use_cluster):
                    op.attributes["rocdl.cluster_dims"] = ir.StringAttr.get(
                        f"{cluster_m},{cluster_n},1")
        cluster_arg = (cluster_m, cluster_n, 1) if use_cluster else None
        launcher.launch(
            grid=(gx, gy, gz),
            block=(block_threads, 1, 1),
            stream=stream,
            cluster=cluster_arg,
        )

    @flyc.jit
    def launch_mxscale_gemm_masked(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_a_scale: fx.Tensor,
        arg_b_scale: fx.Tensor,
        arg_masked_m: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        stream: fx.Stream,
    ):
        _ = cache_tag
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            arena_alloc.finalized = False
            arena_alloc.finalize()

        idx_m = arith.index_cast(T.index, i32_m.ir_value())
        idx_n = arith.index_cast(T.index, i32_n.ir_value())
        gx = _raw((idx_m + arith.index(tile_m - 1)) / arith.index(tile_m))
        if const_expr(batch_count > 1):
            gx = gx * batch_count
        gy = _raw((idx_n + arith.index(tile_n - 1)) / arith.index(tile_n))
        gz = split_k

        launcher = kernel_mxscale_gemm(
            arg_c, arg_a, arg_b, arg_a_scale, arg_b_scale,
            arg_masked_m, arg_masked_m, arg_masked_m, i32_m, i32_n)
        for op in ctx.gpu_module_body.operations:
            if const_expr(hasattr(op, 'attributes') and op.OPERATION_NAME == "gpu.func"):
                if const_expr(effective_waves_per_eu is not None):
                    _wpe = int(effective_waves_per_eu)
                    if const_expr(_wpe >= 1):
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                            ir.IntegerType.get_signless(32), _wpe)
                if const_expr(use_cluster):
                    op.attributes["rocdl.cluster_dims"] = ir.StringAttr.get(
                        f"{cluster_m},{cluster_n},1")
        cluster_arg = (cluster_m, cluster_n, 1) if use_cluster else None
        launcher.launch(
            grid=(gx, gy, gz),
            block=(block_threads, 1, 1),
            stream=stream,
            cluster=cluster_arg,
        )

    @flyc.jit
    def launch_mxscale_gemm_masked_persistent(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_a_scale: fx.Tensor,
        arg_b_scale: fx.Tensor,
        arg_masked_m: fx.Tensor,
        arg_m_tile_prefix: fx.Tensor,
        arg_m_tile_map: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        stream: fx.Stream,
    ):
        _ = cache_tag
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            arena_alloc.finalized = False
            arena_alloc.finalize()

        idx_n = arith.index_cast(T.index, i32_n.ir_value())
        gx = (idx_n + arith.index(tile_n - 1)) / arith.index(tile_n)
        gy = arith.index(_persistent_workers)
        gz = split_k

        launcher = kernel_mxscale_gemm(
            arg_c, arg_a, arg_b, arg_a_scale, arg_b_scale,
            arg_masked_m, arg_m_tile_prefix, arg_m_tile_map, i32_m, i32_n)
        for op in ctx.gpu_module_body.operations:
            if const_expr(hasattr(op, 'attributes') and op.OPERATION_NAME == "gpu.func"):
                if const_expr(effective_waves_per_eu is not None):
                    _wpe = int(effective_waves_per_eu)
                    if const_expr(_wpe >= 1):
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                            ir.IntegerType.get_signless(32), _wpe)
        launcher.launch(
            grid=(gx, gy, gz),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    if expert_sched_mode:
        launch_mxscale_gemm.compile_hints["llvm_options"] = {
            "amdgpu-expert-scheduling-mode": True,
        }
        launch_mxscale_gemm_masked.compile_hints["llvm_options"] = {
            "amdgpu-expert-scheduling-mode": True,
        }
        launch_mxscale_gemm_masked_persistent.compile_hints["llvm_options"] = {
            "amdgpu-expert-scheduling-mode": True,
        }

    if grouped_masked_m:
        return launch_mxscale_gemm_masked_persistent if grouped_persistent_m else launch_mxscale_gemm_masked
    return launch_mxscale_gemm


compile_mxfp4_gemm = lambda **kw: compile_mxscale_gemm(data_format="fp4", **kw)
compile_mxfp8_gemm = lambda **kw: compile_mxscale_gemm(data_format="fp8", **kw)
compile_a8w4_gemm = lambda **kw: compile_mxscale_gemm(data_format="a8w4", **kw)

__all__ = ["compile_mxscale_gemm", "compile_mxfp4_gemm", "compile_mxfp8_gemm",
           "compile_a8w4_gemm"]
