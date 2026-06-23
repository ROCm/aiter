"""Unified MXFP4/MXFP8/A8W4 GEMM kernel for gfx1250.

Supports FP4 (E2M1), FP8 (E4M3) and A8W4 (FP8 activation + FP4 weight),
selected via ``data_format="fp4"|"fp8"|"a8w4"``. Scales are either E8M0
block scales applied in-MMA (``scale_mode="mxscale"``) or per-token/
per-channel fp32 scales applied in the epilogue (``scale_mode="ptpc"``).
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, llvm, scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import (
    arith,
    buffer_ops,
    const_expr,
    gpu,
    idx2crd,
    range_constexpr,
    rocdl,
    tdm_ops,
)
from flydsl.expr.rocdl import cluster
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr, check_smem_capacity
from .gemm_common_gfx1250 import (
    extract_lds_base_idx,
    get_lds_memref,
    lds_load_b32_raw,
    lds_load_b128_raw,
    pipeline_fence,
    pipeline_fence_signal,
    pipeline_fence_wait,
    store_acc_vec8_to_buffer,
    store_acc_vec8_to_lds,
)
from .pipeline_utils import make_tail_plan, tdm_epilogue_fence_threshold_bytes


def _s_prefetch_inst_burst(num_pages: int, page_bytes: int = 4096):
    """gfx1250: prefetch ``num_pages`` × 4 KB of instructions ahead of PC.

    Caller must keep ``num_pages * page_bytes`` within shader bounds; over-reach
    page-faults.
    """
    from flydsl._mlir.dialects import llvm as _llvm

    lines = [
        f"s_prefetch_inst_pc_rel {pg * page_bytes}, null, 31" for pg in range(num_pages)
    ]
    _llvm.inline_asm(None, [], "\n".join(lines), "", has_side_effects=True)


# Common constants
WMMA_M, WMMA_N, WMMA_K = 16, 16, 128
WAVE_SIZE = 32
SCALE_BLOCK = 32
SCALES_PER_WMMA = WMMA_K // SCALE_BLOCK  # 4


def _vec_chunks(n: int):
    """Compile-time split of n contiguous i32 into buffer_load widths (4/2/1)."""
    chunks = []
    done = 0
    while done < n:
        w = 4 if (n - done) >= 4 else (2 if (n - done) >= 2 else 1)
        chunks.append((done, w))
        done += w
    return chunks


LDS_PAD_A_BYTES = 16
LDS_PAD_D_BYTES = 16
LDS_SEGMENT_BYTES = 64 * 1024
LDS_GFX1250_MAX_BYTES = 5 * LDS_SEGMENT_BYTES


@functools.lru_cache(maxsize=256)
def compile_fp8fp4_gemm(
    *,
    data_format: str = "fp4",
    scale_mode: str = "mxscale",
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
    out_dtype: str = "f32",
    inst_prefetch: bool = False,
    split_k: int = 1,
    expert_sched_mode: bool = True,
    atomic_barrier_enable: bool = False,
    ascale_load_path: str = "vgpr",
):
    """Compile an FP4/FP8/A8W4 GEMM kernel with TDM async copy.

    Args:
        data_format: "fp4" (E2M1), "fp8" (E4M3), or "a8w4" (FP8 act + FP4 weight).
        scale_mode: "mxscale" (E8M0 block scale via V_WMMA_SCALE) or "ptpc"
            (per-token sa[M] / per-channel sb[N] fp32, applied in the epilogue).

    Data layout:
        A: [M, K_packed] uint8 (FP4: K_packed=K//2, FP8: K_packed=K)
        B: [N, K_packed] uint8, preshuffled (16x16 byte tiles)
        mxscale scale_A:
            ascale_load_path="vgpr": [M, K//32] uint8 E8M0
            ascale_load_path="shuffled_tdm": [ceil(M/32), (K//128)*128] uint8 E8M0
                in 32x4 packed layout
        mxscale scale_B: [N//32, (K//128)*128] uint8 E8M0 in 32x4 packed layout
        ptpc:    scale_A [M], scale_B [N] fp32

    Returns a JitFunction:
        launch_fn(arg_c, arg_a, arg_b, arg_a_scale, arg_b_scale, M, N, lda, ldc, stream)
    where lda/ldc are A/C runtime leading-dim strides in elements (dense: lda=K, ldc=N).
    """
    if data_format not in ("fp4", "fp8", "a8w4"):
        raise ValueError(
            f"data_format must be 'fp4', 'fp8', or 'a8w4', got {data_format!r}"
        )
    if scale_mode not in ("mxscale", "ptpc"):
        raise ValueError(f"scale_mode must be 'mxscale' or 'ptpc', got {scale_mode!r}")
    if scale_mode == "ptpc" and data_format not in ("fp8", "a8w4"):
        raise ValueError(
            "scale_mode='ptpc' currently only supports data_format='fp8' or 'a8w4'"
        )
    if ascale_load_path not in ("vgpr", "shuffled_tdm"):
        raise ValueError(
            f"ascale_load_path must be 'vgpr' or 'shuffled_tdm', got {ascale_load_path!r}"
        )

    is_fp4 = data_format == "fp4"
    is_a8w4 = data_format == "a8w4"
    is_ptpc = scale_mode == "ptpc"

    if out_dtype not in ("f32", "bf16", "f16"):
        raise ValueError(
            f"out_dtype must be 'f32', 'bf16', or 'f16', got {out_dtype!r}"
        )
    elem_bytes_d = 2 if out_dtype in ("bf16", "f16") else 4
    effective_expert_sched_mode = bool(expert_sched_mode)

    if num_buffers not in (2, 3, 4, 5, 6):
        raise ValueError(f"num_buffers must be 2, 3, 4, 5 or 6, got {num_buffers}")
    if split_k < 1:
        raise ValueError(f"split_k must be >= 1, got {split_k}")
    tdm_store_enabled = split_k == 1

    use_cluster = cluster_m > 1 or cluster_n > 1
    if use_cluster:
        if cluster_m * cluster_n > 16:
            raise ValueError(
                f"cluster_m * cluster_n must be <= 16, got {cluster_m}*{cluster_n}"
            )
    effective_waves_per_eu = waves_per_eu

    num_warps = m_warp * n_warp
    block_threads = num_warps * WAVE_SIZE
    if block_threads > 1024:
        raise ValueError(f"block_threads must be <= 1024, got {block_threads}")

    # ── Format-dependent compile-time constants ──
    # A8W4: activation is FP8 (PACK_FACTOR_A=1), weight is FP4 (PACK_FACTOR_B=2)
    if is_a8w4:
        PACK_FACTOR_A = 1  # FP8 activation
        PACK_FACTOR_B = 2  # FP4 weight
    elif is_fp4:
        PACK_FACTOR_A = 2
        PACK_FACTOR_B = 2
    else:
        PACK_FACTOR_A = 1
        PACK_FACTOR_B = 1

    WMMA_N_EFF = 32 if is_fp4 else 16  # N-cols covered per WMMA instruction
    ACC_VEC_SIZE = 16 if is_fp4 else 8  # accumulator vector width
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
            f"K/split_k must be divisible by tile_k={tile_k}, got {split_k_chunk}"
        )
    if tile_k % WMMA_K != 0:
        raise ValueError(f"tile_k must be a multiple of {WMMA_K}, got {tile_k}")
    if tile_m % WMMA_M != 0:
        raise ValueError(f"tile_m must be a multiple of {WMMA_M}, got {tile_m}")
    if tile_n % WMMA_N != 0:
        raise ValueError(f"tile_n must be a multiple of {WMMA_N}, got {tile_n}")
    if packed_tile_k_a % 4 != 0:
        raise ValueError(
            f"packed_tile_k_a must be a multiple of 4, got {packed_tile_k_a}"
        )
    if packed_tile_k_b % 4 != 0:
        raise ValueError(
            f"packed_tile_k_b must be a multiple of 4, got {packed_tile_k_b}"
        )
    if scale_k_per_tile % 4 != 0:
        raise ValueError(
            f"scale_k_per_tile must be a multiple of 4 (tile_k >= 128), got {scale_k_per_tile}"
        )

    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp
    if warp_tile_m % WMMA_M != 0:
        raise ValueError(f"warp_tile_m={warp_tile_m} must be a multiple of {WMMA_M}")
    if warp_tile_n % WMMA_N_EFF != 0:
        raise ValueError(
            f"warp_tile_n={warp_tile_n} must be a multiple of {WMMA_N_EFF}"
        )

    # mxscale B-scale is always the 32x4 `preshuffle_scale` layout: require N/tile_n a
    # multiple of 32 and tile_k a multiple of 128 (no legacy sub-32 fallback).
    if scale_mode == "mxscale" and (
        N % 32 != 0 or tile_n % 32 != 0 or tile_k % 128 != 0
    ):
        raise ValueError(
            f"mxscale 32x4 B-scale requires N%32==0, tile_n%32==0, tile_k%128==0; "
            f"got N={N}, tile_n={tile_n}, tile_k={tile_k}"
        )

    num_k_tiles = split_k_chunk // tile_k
    if num_k_tiles < num_buffers:
        raise ValueError(
            f"{num_buffers}-stage buffering requires num_k_tiles >= {num_buffers}, got {num_k_tiles}"
        )

    gpu_arch = str(get_hip_arch())
    assert gpu_arch.startswith("gfx1250"), f"Expected gfx1250, got {gpu_arch}"

    k_wmma_steps = tile_k // WMMA_K

    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N_EFF
    n_accs = wmma_m_rep * wmma_n_rep
    # FP4 A/B swap: BScale rep derived from WMMA_M, not WMMA_N_EFF
    b_scale_load_rep = warp_tile_n // WMMA_M if is_fp4 else wmma_n_rep

    # mxscale carries per-K-block scales; ptpc has no K-loop scale (per-token/
    # per-channel fp32 applied in the epilogue).
    is_mxscale = not is_ptpc
    use_ascale_vgpr = is_mxscale and ascale_load_path == "vgpr"
    use_ascale_shuffled_tdm = is_mxscale and ascale_load_path == "shuffled_tdm"

    # 32x4 A-scale layout (preshuffle_scale): [ceil(M/32), K//128, 32, 4].
    # One 128B block (32 rows x 4 K-scales) maps to one WMMA scale operand.
    as32_block_bytes = 128
    as32_global_row_stride = 0
    as32_lds_row_stride = 0
    as32_tile_blocks_pad = 1
    as32_n_load = 0
    as32_opsel = False
    # 32x4 B-scale layout (preshuffle_scale): [N//32, K//128, 32, 4].
    bs32_block_bytes = 128
    bs32_global_row_stride = 0
    bs32_lds_row_stride = 0
    bs32_tile_blocks_pad = 1
    bs32_n_load = 0
    bs32_opsel = False
    if is_mxscale:
        if use_ascale_shuffled_tdm:
            as32_global_row_stride = (K // WMMA_K) * as32_block_bytes
            as32_lds_row_stride = k_wmma_steps * as32_block_bytes
            as32_tile_blocks = (tile_m + 31) // 32
            as32_tile_blocks_pad = 1 << (as32_tile_blocks - 1).bit_length()
            # Adjacent 16-M WMMAs share one 32-row block when the warp M span is even.
            as32_opsel = wmma_m_rep >= 2 and (wmma_m_rep % 2 == 0)
            as32_n_load = (wmma_m_rep // 2) if as32_opsel else wmma_m_rep

        bs32_global_row_stride = (
            K // WMMA_K
        ) * bs32_block_bytes  # bytes per block row (= K)
        bs32_lds_row_stride = k_wmma_steps * bs32_block_bytes  # LDS bytes per block row
        bs32_tile_blocks = tile_n // 32
        # Pad block count to pow2 so the TDM warp split stays clean (non-pow2, e.g.
        # 6, miscopies LDS). Cost-free for pow2 block counts; else 1-2 oob-clipped.
        bs32_tile_blocks_pad = 1 << (bs32_tile_blocks - 1).bit_length()
        bs32_opsel = (not is_fp4) and (wmma_n_rep % 2 == 0)
        bs32_n_load = (
            (wmma_n_rep // 2) if bs32_opsel else wmma_n_rep
        )  # b32 loads per ks

    # A-scale VGPR path keeps the original [M, K//32] layout. Its op_sel pairing is
    # by M-half because lane_kgrp selects the upper/lower half of the warp's M span.
    ascale_opsel = (
        use_ascale_vgpr and wmma_m_rep >= 2 and (wmma_m_rep & (wmma_m_rep - 1)) == 0
    )
    ascale_half = wmma_m_rep // 2
    ascale_load = ascale_half if ascale_opsel else wmma_m_rep

    # TDM loader assignment:
    #   VGPR A-scale: wave0=A, wave1=B, wave2=B-scale; at 2 waves B-scale rides wave0.
    #   Shuffled A-scale: wave0=A, wave1=B, wave2=A-scale, wave3=B-scale; with 2/3
    #   waves the missing scale descriptor rides as a secondary issue.
    two_wave_bscale = use_ascale_vgpr and num_warps == 2
    two_wave_scale = use_ascale_shuffled_tdm and num_warps == 2
    three_wave_bscale = use_ascale_shuffled_tdm and num_warps == 3
    secondary_scale_tdm = two_wave_bscale or two_wave_scale or three_wave_bscale

    # mxscale uses at least A/B TDM waves; ptpc uses A/B only.
    if num_warps < 2:
        raise ValueError(
            f"wave-specialized TDM requires at least 2 waves, got {num_warps}"
        )

    _b_frag_loads_per_wn = 2 if is_a8w4 else 4
    _a_frag_loads_per_wm = 2 if is_fp4 else 4
    # Scale ds_loads issued alongside A/B fragment loads in the streaming schedule
    # (for the partial-drain s_wait_dscnt bookkeeping).
    _a_scale_ds = as32_n_load if use_ascale_shuffled_tdm else 0
    _b_scale_ds = bs32_n_load if is_mxscale else 0
    _scale_ds_loads = _a_scale_ds + _b_scale_ds
    _a_frag_ds = wmma_m_rep * _a_frag_loads_per_wm
    _bs_ds_loads = wmma_n_rep * _b_frag_loads_per_wn + _scale_ds_loads
    _as_ds_loads = _a_frag_ds + _scale_ds_loads
    _row_major_k_prefetch_bundle_ds = _a_frag_ds + _bs_ds_loads

    lds_a_stride_bytes = packed_tile_k_a + LDS_PAD_A_BYTES

    lds_a_data_bytes = tile_m * lds_a_stride_bytes
    lds_b_data_bytes = tile_n * packed_tile_k_b
    _scale_guard_bytes = 16
    # A-scale LDS is allocated only for the shuffled TDM path.
    lds_a_scale_bytes = (
        (as32_tile_blocks_pad * as32_lds_row_stride + _scale_guard_bytes)
        if use_ascale_shuffled_tdm
        else 0
    )
    lds_b_scale_bytes = (
        (bs32_tile_blocks_pad * bs32_lds_row_stride + _scale_guard_bytes)
        if is_mxscale
        else 0
    )

    def _align_up(value: int, align: int) -> int:
        if value % align == 0:
            return value
        return (value + align - 1) // align * align

    # TDM descriptors partition a tile cooperatively across ``num_warps`` by
    # deriving per-wave offsets from ``wave_id``. In wave-specialized mode we
    # dedicate one loader wave to each tensor (A/B/A_scale/B_scale), so each
    # active loader wave must issue a full-tile descriptor by itself.
    tdm_desc_num_warps = 1

    # All pipeline stages share the same intra-stage layout in the generic
    # arena path. The active gfx1250 FP8 TDM tile uses a separate reference
    # pool layout below.
    stage_layout = SmemAllocator(
        None, arch=gpu_arch, global_sym_name=f"mxscale_{data_format}_layout"
    )
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
            f"mxscale_{data_format}_{tile_m}x{tile_n}x{tile_k}_{m_warp}x{n_warp}_{num_buffers}buf_arena"
        ),
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

    if tdm_store_enabled:
        lds_d_row_stride = warp_tile_n * elem_bytes_d + LDS_PAD_D_BYTES
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

    # TENSORcnt is tracked per-wave in hardware. Keep the fence budget in stage units;
    # secondary scale descriptors on 2/3-wave mxscale paths only make this more conservative.
    TDM_LOADS_PER_STEP = 1
    tail_plan = [
        (ls, cs, o * TDM_LOADS_PER_STEP // 2 if o > 0 else o)
        for ls, cs, o in _base_tail_plan
    ]

    # Pre-compute epilogue sub-tile layout (unified for FP4 vec16 and FP8 vec8)
    _sub_tiles = []
    for _wm in range(wmma_m_rep):
        for _wn in range(wmma_n_rep):
            if is_fp4:
                # vec<16,f32>: split into 2 × 8 elements (2 × 16-col halves)
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
    COMPUTE_SCHEDULE_FP4_QUADRANT = "fp4_quadrant"
    COMPUTE_SCHEDULE_FP8_QUADRANT = "fp8_quadrant"
    COMPUTE_SCHEDULE_FP8_DEEP_PIPELINE = "fp8_deep_pipeline"

    fp8_deep_pipeline_eligible = (
        data_format in ("fp8", "a8w4")
        and tile_m == 256
        and tile_n == 256
        and tile_k == 128
        and m_warp == 2
        and n_warp == 2
        and num_buffers == 4
        and out_dtype == "bf16"
    )

    def _pick_compute_schedule_kind():
        if wmma_m_rep % 2 != 0 or wmma_n_rep % 2 != 0 or n_accs < 8:
            return COMPUTE_SCHEDULE_ROW_MAJOR_STREAMING
        # Quadrant: split B left/right, compute the 4 quadrants to widen the
        # LDS-load-to-WMMA distance. FP4/FP8 differ only in per-format wait tuning.
        if is_fp4:
            return COMPUTE_SCHEDULE_FP4_QUADRANT
        # A8W4 (FP8 act + FP4 weight) shares FP8's accumulator layout and operand
        # path, so it reuses the FP8 schedules.
        if data_format in ("fp8", "a8w4"):
            if fp8_deep_pipeline_eligible:
                return COMPUTE_SCHEDULE_FP8_DEEP_PIPELINE
            return COMPUTE_SCHEDULE_FP8_QUADRANT
        return COMPUTE_SCHEDULE_ROW_MAJOR_STREAMING

    compute_schedule_kind = _pick_compute_schedule_kind()
    use_row_major_streaming_schedule = (
        compute_schedule_kind == COMPUTE_SCHEDULE_ROW_MAJOR_STREAMING
    )
    use_fp4_quadrant_schedule = compute_schedule_kind == COMPUTE_SCHEDULE_FP4_QUADRANT
    use_fp8_quadrant_schedule = compute_schedule_kind == COMPUTE_SCHEDULE_FP8_QUADRANT
    use_fp8_deep_pipeline_schedule = (
        compute_schedule_kind == COMPUTE_SCHEDULE_FP8_DEEP_PIPELINE
    )
    use_row_major_k_prefetch = wmma_m_rep == 1 and k_wmma_steps > 1
    _row_major_k_prefetch_depth = 2 if use_row_major_k_prefetch else 1
    _row_major_k_prefetch_depth = max(
        0, min(k_wmma_steps - 1, _row_major_k_prefetch_depth)
    )
    use_row_major_late_signal = use_row_major_k_prefetch

    # A-scale VGPR-ring prefetch depth (K-tiles ahead).  Deeper K tiles expose
    # more latency to hide; depth 4 improves the small-M row-major large-K path
    if use_ascale_vgpr and use_row_major_streaming_schedule:
        _bvs_D = 4 if num_buffers >= 4 else 3
    else:
        _bvs_D = 1
    _bvs_active = use_ascale_vgpr

    if is_mxscale:
        assert compute_schedule_kind in (
            COMPUTE_SCHEDULE_ROW_MAJOR_STREAMING,
            COMPUTE_SCHEDULE_FP8_QUADRANT,
            COMPUTE_SCHEDULE_FP8_DEEP_PIPELINE,
            COMPUTE_SCHEDULE_FP4_QUADRANT,
        )
    use_ws_tdm_split_signal_overlap = (
        (use_fp8_quadrant_schedule or use_fp8_deep_pipeline_schedule)
        and num_buffers == 4
        and use_cluster
    )
    use_tdm_late_signal_overlap = (
        use_ws_tdm_split_signal_overlap or use_row_major_late_signal
    )

    if use_fp4_quadrant_schedule:
        _fp4_half_wm = wmma_m_rep // 2
        _fp4_half_wn = wmma_n_rep // 2
        _fp4_group_size = _fp4_half_wm * _fp4_half_wn

    if use_fp8_quadrant_schedule or use_fp8_deep_pipeline_schedule:
        _fp8_half_wm = wmma_m_rep // 2
        _fp8_half_wn = wmma_n_rep // 2
        _fp8_group_size = _fp8_half_wm * _fp8_half_wn
        if is_mxscale:
            _fp8_b_scale_loads = bs32_n_load  # 32x4: one b32 per block-or-WMMA per ks
        else:
            _fp8_b_scale_loads = 0 if is_ptpc else (b_scale_load_rep + 3) // 4
    if use_fp8_deep_pipeline_schedule:
        _fp8_pair_wm = 2
        _fp8_pair_wn = 2
        _fp8_wm_pairs = wmma_m_rep // _fp8_pair_wm
        _fp8_wn_pairs = wmma_n_rep // _fp8_pair_wn
        _fp8_pair_a_loads = _fp8_pair_wm * DS_LOADS_PER_A_FRAG
        _fp8_pair_b_loads = _fp8_pair_wn * _b_frag_loads_per_wn
        # Scale ds_loads issued at the loop top. Uses the finalized module-level counts.
        _fp8_scale_loads = 0 if is_ptpc else (_a_scale_ds + _b_scale_ds)

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def kernel_mxscale_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_a_scale: fx.Tensor,
        arg_b_scale: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        i32_lda: fx.Int32,
        i32_ldc: fx.Int32,
    ):
        # Enable back-to-back WMMA issue (SCHED_MODE bit[4] = DISABLE_VALU_STALL)
        rocdl.disable_xdl_arb_stall()

        if const_expr(inst_prefetch):
            if rocdl.wave_id() == fx.Int32(0):
                _s_prefetch_inst_burst(num_pages=4)

        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        by = gpu.block_id("y")
        bz = fx.Index(gpu.block_idx.z) if split_k > 1 else arith.index(0)

        blk_m = bx * arith.index(tile_m)
        blk_n = by * arith.index(tile_n)
        split_k_base = bz * arith.index(split_k_chunk)

        if const_expr(use_cluster):
            local_x, local_y = cluster.compute_cluster_position()
            a_mcast_mask, b_mcast_mask = cluster.compute_mcast_masks(
                local_x, local_y, cluster_m, cluster_n
            )
        else:
            a_mcast_mask = 0
            b_mcast_mask = 0

        # The FP8 deep pipeline runs cleaner when adjacent wave ids advance M
        # first; keep the default mapping for the other schedules.
        if const_expr(use_fp8_deep_pipeline_schedule):
            layout_thr = fx.make_layout(
                (m_warp, n_warp, 2, 16), (WAVE_SIZE, m_warp * WAVE_SIZE, 16, 1)
            )
        else:
            layout_thr = fx.make_layout(
                (m_warp, n_warp, 2, 16), (n_warp * WAVE_SIZE, WAVE_SIZE, 16, 1)
            )
        thr_coord = idx2crd(fx.Int32(tx), layout_thr)
        wave_m_idx, wave_n_idx, lane_kgrp, lane16 = (
            fx.get(thr_coord, 0),
            fx.get(thr_coord, 1),
            fx.get(thr_coord, 2),
            fx.get(thr_coord, 3),
        )

        warp_m_base = wave_m_idx * arith.index(warp_tile_m)
        warp_n_base = wave_n_idx * arith.index(warp_tile_n)
        m_idx = fx.Index(i32_m)

        def _load_contig_i32(rsrc, base_idx, n, soff):
            # Load n contiguous i32 values through the widest legal buffer_load chunks.
            out = [None] * n
            _chunks = _vec_chunks(n)
            for _ci in range_constexpr(len(_chunks)):
                start, w = _chunks[_ci]
                off = arith.index_cast(T.i32, base_idx + arith.index(start))
                r = buffer_ops.buffer_load(
                    rsrc, off, vec_width=w, dtype=T.i32, soffset_bytes=soff
                )
                if const_expr(w == 1):
                    out[start] = r
                else:
                    rv = fx.Vector(r)
                    for c in range_constexpr(w):
                        out[start + c] = rv[c]
            return out

        _scale_identity_i32 = arith.constant(0x7F7F7F7F, type=T.i32)

        if const_expr(use_ascale_vgpr):
            # A-scale VGPR path: read scale_A[M, K//32] directly from its row-major layout.
            _ascale_nbytes = m_idx * arith.index(K_scale)
            _ascale_rsrc = buffer_ops.create_buffer_resource(
                arg_a_scale,
                max_size=False,
                num_records_bytes=_ascale_nbytes,
            )
            _ascale_row_i32 = K_scale // 4
            _ascale_row0 = blk_m + warp_m_base + lane16
            if const_expr(ascale_opsel):
                _ascale_row0 = _ascale_row0 + lane_kgrp * arith.index(
                    ascale_half * WMMA_M
                )
            _vs_tile_a = k_wmma_steps * ascale_load

            def _load_contig_i32_guarded_row(row, n, soff):
                row_valid = row < m_idx
                if_op = scf.IfOp(row_valid, [T.i32] * n, has_else=True)
                with ir.InsertionPoint(if_op.then_block):
                    vals = _load_contig_i32(
                        _ascale_rsrc,
                        row * arith.index(_ascale_row_i32),
                        n,
                        soff,
                    )
                    scf.YieldOp([arith.unwrap(v) for v in vals])
                with ir.InsertionPoint(if_op.else_block):
                    scf.YieldOp([arith.unwrap(_scale_identity_i32) for _ in range(n)])
                return list(if_op.results)

            def _load_ascale_impl(k_base, guarded):
                kt = k_base // arith.index(tile_k)
                soff = arith.index_cast(T.i32, kt * arith.index(scale_k_per_tile))
                vals = [None] * (k_wmma_steps * ascale_load)
                for i in range_constexpr(ascale_load):
                    row = _ascale_row0 + arith.index(i * WMMA_M)
                    if const_expr(guarded):
                        ks_vals = _load_contig_i32_guarded_row(row, k_wmma_steps, soff)
                    else:
                        vidx = row * arith.index(_ascale_row_i32)
                        ks_vals = _load_contig_i32(
                            _ascale_rsrc, vidx, k_wmma_steps, soff
                        )
                    for ks in range_constexpr(k_wmma_steps):
                        vals[ks * ascale_load + i] = ks_vals[ks]
                return vals

            def _load_ascale(k_base):
                full_tile = (blk_m + arith.index(tile_m)) <= m_idx
                if_op = scf.IfOp(full_tile, [T.i32] * _vs_tile_a, has_else=True)
                with ir.InsertionPoint(if_op.then_block):
                    scf.YieldOp(
                        [
                            arith.unwrap(v)
                            for v in _load_ascale_impl(k_base, guarded=False)
                        ]
                    )
                with ir.InsertionPoint(if_op.else_block):
                    scf.YieldOp(
                        [
                            arith.unwrap(v)
                            for v in _load_ascale_impl(k_base, guarded=True)
                        ]
                    )
                return list(if_op.results)

            _bvs_prefetch = _load_ascale

        # Runtime leading-dim strides (strided A/C). Dense callers pass lda == K,
        # ldc == N for byte-identical addressing. A's stride is in packed elements.
        if const_expr(PACK_FACTOR_A == 1):
            lda_packed = fx.Index(i32_lda)
        else:
            lda_packed = fx.Index(i32_lda) / arith.index(PACK_FACTOR_A)

        n_stride = fx.Index(i32_ldc)
        c_nrec = m_idx * n_stride * arith.index(elem_bytes_d)
        c_rsrc = buffer_ops.create_buffer_resource(arg_c, num_records_bytes=c_nrec)
        c_global_ptr_type = ir.Type.parse("!llvm.ptr<1>")
        c_global_base_i64 = llvm.PtrToIntOp(
            T.i64,
            fly.extract_aligned_pointer_as_index(
                c_global_ptr_type, arg_c.__extract_to_ir_values__()[0]
            ),
        ).result

        def make_desc_a(memref, k_base):
            k_packed_off = k_base // arith.index(PACK_FACTOR_A)
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_a,
                lds_memref=memref,
                global_offset=(blk_m, k_packed_off),
                tensor_shape=(tile_m, packed_tile_k_a),
                strides=(lda_packed, 1),
                tile_shape=(tile_m, packed_tile_k_a),
                elem_bytes=1,
                pad_interval=packed_tile_k_a,
                pad_amount=LDS_PAD_A_BYTES,
                num_warps=tdm_desc_num_warps,
                workgroup_mask=a_mcast_mask,
                atomic_barrier_enable=atomic_barrier_enable,
                early_timeout=True,
                oob_outer_bound=i32_m,
            )

        def make_desc_b(memref, k_base):
            k_packed_off = k_base // arith.index(PACK_FACTOR_B)
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_b,
                lds_memref=memref,
                global_offset=(
                    blk_n // arith.index(16),
                    k_packed_off * arith.index(16),
                ),
                tensor_shape=(N // 16, K_packed_b * 16),
                strides=(K_packed_b * 16, 1),
                tile_shape=(tile_n // 16, packed_tile_k_b * 16),
                elem_bytes=1,
                pad_interval=0,
                pad_amount=0,
                num_warps=tdm_desc_num_warps,
                workgroup_mask=b_mcast_mask,
                atomic_barrier_enable=atomic_barrier_enable,
                early_timeout=True,
            )

        def make_desc_bs(memref, k_base):
            # 32x4: copy this tile's 32-N blocks x K-blocks slice of the preshuffled
            # [N//32, (K//128)*128] B-scale tensor.
            block_off = blk_n // arith.index(32)
            col_off = (k_base // arith.index(WMMA_K)) * arith.index(bs32_block_bytes)
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_b_scale,
                lds_memref=memref,
                global_offset=(block_off, col_off),
                tensor_shape=(N // 32, bs32_global_row_stride),
                strides=(bs32_global_row_stride, 1),
                tile_shape=(bs32_tile_blocks_pad, bs32_lds_row_stride),
                elem_bytes=1,
                pad_interval=0,
                pad_amount=0,
                num_warps=tdm_desc_num_warps,
                workgroup_mask=b_mcast_mask,
                atomic_barrier_enable=atomic_barrier_enable,
                early_timeout=True,
                oob_outer_bound=N // 32,
            )

        def make_desc_as(memref, k_base):
            # 32x4: copy this tile's M block rows from the packed A-scale tensor.
            # Runtime OOB clips whole missing block rows; the LDS reader masks lanes
            # inside the final partial block to the E8M0 identity value.
            block_off = blk_m // arith.index(32)
            col_off = (k_base // arith.index(WMMA_K)) * arith.index(as32_block_bytes)
            m_block_bound = (m_idx + arith.index(31)) // arith.index(32)
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_a_scale,
                lds_memref=memref,
                global_offset=(block_off, col_off),
                tensor_shape=(as32_tile_blocks_pad, as32_global_row_stride),
                strides=(as32_global_row_stride, 1),
                tile_shape=(as32_tile_blocks_pad, as32_lds_row_stride),
                elem_bytes=1,
                pad_interval=0,
                pad_amount=0,
                num_warps=tdm_desc_num_warps,
                workgroup_mask=a_mcast_mask,
                atomic_barrier_enable=atomic_barrier_enable,
                early_timeout=True,
                oob_outer_bound=m_block_bound,
            )

        tdm_wave_id = rocdl.wave_id()
        tdm_wave_is_a = tdm_wave_id == fx.Int32(0)
        tdm_wave_is_b = tdm_wave_id == fx.Int32(1)
        tdm_wave_is_as = tdm_wave_id == fx.Int32(2)

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
                base = (
                    row_base
                    + arith.index(wm * WMMA_M * lds_a_stride_bytes)
                    + k_half_off
                )
                bases.append(base)
            return lds_ptr, bases

        def load_a_frag(lds_buffer, a_lane_base, ks):
            """Load one A-fragment from LDS.

            FP4: vec<8xi32> via 2 × ds_load_b128 (32 bytes per lane).
            FP8/A8W4: vec<16xi32> via 4 × ds_load_b128 (64 bytes per lane).
              Interleaved K layout:
              kgrp0 reads bytes [0:15],[32:47],[64:79],[96:111] (stride=32)
              kgrp1 reads bytes [16:31],[48:63],[80:95],[112:127] (stride=32)
            """
            k_byte_off = arith.index(ks * WMMA_K // PACK_FACTOR_A)
            byte_off = a_lane_base + k_byte_off
            v0 = fx.Vector(lds_load_b128_raw(lds_buffer, byte_off))
            if const_expr(is_fp4):
                # Interleaved stride=32: +0, +32
                v1 = fx.Vector(
                    lds_load_b128_raw(lds_buffer, byte_off + arith.index(32))
                )
                return v0.shuffle(v1, list(range(8)))
            else:
                # Interleaved stride=32: +0, +32, +64, +96
                v1 = fx.Vector(
                    lds_load_b128_raw(lds_buffer, byte_off + arith.index(32))
                )
                v2 = fx.Vector(
                    lds_load_b128_raw(lds_buffer, byte_off + arith.index(64))
                )
                v3 = fx.Vector(
                    lds_load_b128_raw(lds_buffer, byte_off + arith.index(96))
                )
                v01 = v0.shuffle(v1, list(range(8)))
                v23 = v2.shuffle(v3, list(range(8)))
                return v01.shuffle(v23, list(range(16)))

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
            # All formats: interleaved — kgrp offset = 1 tile = 256 bytes
            k_tile_off = lane_kgrp * arith.index(256)
            bases = []
            if const_expr(is_fp4):
                for wn_half in range_constexpr(wmma_n_rep * 2):
                    ngroup_off = _n_group_base * arith.index(
                        _ngroup_stride
                    ) + arith.index(wn_half * _ngroup_stride)
                    bases.append(ngroup_off + row_off + k_tile_off)
            else:
                # FP8 and A8W4: 1 base per wn (16-col WMMA)
                for wn in range_constexpr(wmma_n_rep):
                    ngroup_off = _n_group_base * arith.index(
                        _ngroup_stride
                    ) + arith.index(wn * _ngroup_stride)
                    bases.append(ngroup_off + row_off + k_tile_off)
            return lds_ptr, bases

        def load_b_frag(lds_buffer, b_lane_bases, wn, ks):
            """Load one B-fragment from preshuffled LDS.

            FP4: 32x128 → vec<16xi32> from 2 N-groups (bases[wn*2], bases[wn*2+1]).
            FP8: 16x128 → vec<16xi32> from 1 N-group (bases[wn]).
            A8W4: 16x128 FP4 → vec<8xi32> from 1 N-group (bases[wn]).

            K-dimension interleaving (FP8/A8W4):
              Stride = 2 tiles = 512 bytes between loads.
              kgrp0 reads tiles 0,2,4,6; kgrp1 reads tiles 1,3,5,7.
            """
            if const_expr(is_fp4):
                # FP4: 2 N-groups per wn, 4 tiles per N-group
                # Interleaved stride=512 (2 tiles): kgrp0→tiles 0,2; kgrp1→tiles 1,3
                _num_tiles = WMMA_K // PACK_FACTOR_B // 16  # 4 tiles total per N-group
                k_subtile_off = arith.index(ks * _num_tiles * 256)
                base0 = b_lane_bases[wn * 2] + k_subtile_off
                v0 = fx.Vector(lds_load_b128_raw(lds_buffer, base0))
                v1 = fx.Vector(lds_load_b128_raw(lds_buffer, base0 + arith.index(512)))
                base1 = b_lane_bases[wn * 2 + 1] + k_subtile_off
                v2 = fx.Vector(lds_load_b128_raw(lds_buffer, base1))
                v3 = fx.Vector(lds_load_b128_raw(lds_buffer, base1 + arith.index(512)))
                v01 = v0.shuffle(v1, list(range(8)))
                v23 = v2.shuffle(v3, list(range(8)))
                return v01.shuffle(v23, list(range(16)))
            elif const_expr(is_a8w4):
                # A8W4: FP4 weight, 4 tiles per N-group
                # Interleaved stride=512: kgrp0→tiles 0,2; kgrp1→tiles 1,3
                _num_tiles = WMMA_K // PACK_FACTOR_B // 16  # 4 tiles total
                k_subtile_off = arith.index(ks * _num_tiles * 256)
                base0 = b_lane_bases[wn] + k_subtile_off
                v0 = fx.Vector(lds_load_b128_raw(lds_buffer, base0))
                v1 = fx.Vector(lds_load_b128_raw(lds_buffer, base0 + arith.index(512)))
                return v0.shuffle(v1, list(range(8)))
            else:
                # FP8: 8 tiles per N-group
                # Interleaved stride=512: kgrp0→tiles 0,2,4,6; kgrp1→tiles 1,3,5,7
                _num_tiles = WMMA_K // PACK_FACTOR_B // 16  # 8 tiles total
                k_subtile_off = arith.index(ks * _num_tiles * 256)
                base0 = b_lane_bases[wn] + k_subtile_off
                v0 = fx.Vector(lds_load_b128_raw(lds_buffer, base0))
                v1 = fx.Vector(lds_load_b128_raw(lds_buffer, base0 + arith.index(512)))
                v2 = fx.Vector(lds_load_b128_raw(lds_buffer, base0 + arith.index(1024)))
                v3 = fx.Vector(lds_load_b128_raw(lds_buffer, base0 + arith.index(1536)))
                v01 = v0.shuffle(v1, list(range(8)))
                v23 = v2.shuffle(v3, list(range(8)))
                return v01.shuffle(v23, list(range(16)))

        def _precompute_bs32_bases(lds_ptr):
            """Tile-local 32-N block base for the warp's 32x4 B-scale read.

            An LDS block row (32 N-rows x 4 K-scales = 128B) is one 32-lane WMMA scale
            operand. op_sel path (even rep): the warp owns whole blocks block0+j. Else
            (fp4 / odd rep): each WMMA reads its own 16/32-N into the operand lanes.
            """
            return lds_ptr, warp_n_base // arith.index(32)

        def _precompute_as32_bases(lds_ptr):
            """Tile-local first A row, relative to the copied 32-row block base."""
            return lds_ptr, (blk_m % arith.index(32)) + warp_m_base

        def _mask_a_scale_oob(word, row_abs):
            return arith.select(row_abs < m_idx, word, _scale_identity_i32)

        def _load_scale32_full_blocks(
            lds_buffer,
            block0,
            ks,
            row_stride_bytes,
            block_bytes,
            load_count,
            row_abs0=None,
        ):
            stride = arith.index(row_stride_bytes)
            ks_off = arith.index(ks * block_bytes)
            lane32 = lane_kgrp * arith.index(16) + lane16
            lane = lane32 * arith.index(4)
            results = []
            for i in range_constexpr(load_count):
                off = (block0 + arith.index(i)) * stride + ks_off + lane
                word = lds_load_b32_raw(lds_buffer, off)
                if const_expr(row_abs0 is not None):
                    word = _mask_a_scale_oob(
                        word, row_abs0 + arith.index(i * 32) + lane32
                    )
                results.append(word)
            return results

        def _load_scale32_half_blocks(
            lds_buffer,
            row16_base,
            ks,
            row_stride_bytes,
            block_bytes,
            load_count,
            row_abs_base=None,
        ):
            stride = arith.index(row_stride_bytes)
            ks_off = arith.index(ks * block_bytes)
            results = []
            for i in range_constexpr(load_count):
                row16 = row16_base + arith.index(i * 16)
                off = (
                    (row16 // arith.index(32)) * stride
                    + ks_off
                    + (row16 % arith.index(32) + lane16) * arith.index(4)
                )
                word = lds_load_b32_raw(lds_buffer, off)
                if const_expr(row_abs_base is not None):
                    word = _mask_a_scale_oob(
                        word, row_abs_base + arith.index(i * 16) + lane16
                    )
                results.append(word)
            return results

        def load_as32_ascale(lds_buffer, row0, ks):
            """Load 32x4 A-scale i32s for K-subtile *ks*."""
            if const_expr(as32_opsel):
                return _load_scale32_full_blocks(
                    lds_buffer,
                    row0 // arith.index(32),
                    ks,
                    as32_lds_row_stride,
                    as32_block_bytes,
                    wmma_m_rep // 2,
                    row_abs0=blk_m + warp_m_base,
                )
            return _load_scale32_half_blocks(
                lds_buffer,
                row0,
                ks,
                as32_lds_row_stride,
                as32_block_bytes,
                wmma_m_rep,
                row_abs_base=blk_m + warp_m_base,
            )

        def load_bs32_bscale(lds_buffer, block0, ks):
            """Load 32x4 B-scale i32s for K-subtile *ks* (one b32 per block-or-WMMA)."""
            if const_expr(bs32_opsel):
                # Even rep: full 32-lane block; op_sel picks the 16-half in _emit_wmma.
                return _load_scale32_full_blocks(
                    lds_buffer,
                    block0,
                    ks,
                    bs32_lds_row_stride,
                    bs32_block_bytes,
                    wmma_n_rep // 2,
                )
            elif const_expr(is_fp4):
                # fp4: one 32-N block per WMMA (no op_sel).
                return _load_scale32_full_blocks(
                    lds_buffer,
                    block0,
                    ks,
                    bs32_lds_row_stride,
                    bs32_block_bytes,
                    wmma_n_rep,
                )
            # fp8 odd rep: each WMMA's 16-N into lanes 0-15 (op_sel=0); the block
            # and its 16-half are runtime (warp may start mid-block).
            return _load_scale32_half_blocks(
                lds_buffer,
                warp_n_base,
                ks,
                bs32_lds_row_stride,
                bs32_block_bytes,
                wmma_n_rep,
            )

        def _load_a_scale_lds(as_buf, as_row0, ks):
            """Load 32x4 A-scale from LDS (mxscale only)."""
            return load_as32_ascale(as_buf, as_row0, ks)

        # Current tile's VGPR-path A-scales, ordered [k_wmma_step][M-rep].
        _vgpr_scale_box = [None]

        def _set_vgpr_a_scales(scale_k_base, pf_a_scales):
            if const_expr(use_ascale_vgpr):
                if const_expr(pf_a_scales is not None):
                    _vgpr_scale_box[0] = pf_a_scales
                else:
                    rocdl.sched_barrier(0)
                    _vgpr_scale_box[0] = _bvs_prefetch(scale_k_base)

        def _load_a_scale_vgpr(ks):
            pf_a = _vgpr_scale_box[0]
            return pf_a[ks * ascale_load : (ks + 1) * ascale_load]

        def _load_b_scale_lds(bs_buf, bs_block0, ks):
            """Load 32x4 B-scale from LDS (mxscale only; ptpc reads no K-loop B-scale)."""
            return load_bs32_bscale(bs_buf, bs_block0, ks)

        def _load_a_scale_operand(as_buf, as_bases, ks):
            if const_expr(use_ascale_vgpr):
                return _load_a_scale_vgpr(ks)
            return _load_a_scale_lds(as_buf, as_bases, ks)

        def _scales_for_emit(as_buf, as_bases, bs_buf, bs_bases, ks):
            """Load scale operands for K-subtile *ks*."""
            if const_expr(is_ptpc):
                return None, None
            a = _load_a_scale_operand(as_buf, as_bases, ks)
            b = _load_b_scale_lds(bs_buf, bs_bases, ks)
            return a, b

        def _load_b_and_scales(b_buf, b_bases, as_buf, as_bases, bs_buf, bs_bases, ks):
            b_frags = [
                load_b_frag(b_buf, b_bases, wn, ks)
                for wn in range_constexpr(wmma_n_rep)
            ]
            a_scales, b_scales = _scales_for_emit(
                as_buf, as_bases, bs_buf, bs_bases, ks
            )
            return b_frags, b_scales, a_scales

        def _emit_wmma(accs, wm, wn, a_frag, b_frag, a_scales, b_scales):
            """Emit one WMMA instruction (format-specific)."""
            idx = wm * wmma_n_rep + wn
            if const_expr(is_ptpc):
                if const_expr(is_a8w4):
                    accs[idx] = rocdl.wmma_scale_f32_16x16x128_f8f6f4(
                        T.vec(8, T.f32),
                        b_frag,
                        a_frag,
                        accs[idx],
                        0x7F7F7F7F,
                        0x7F7F7F7F,
                        fmtA=4,
                        fmtB=0,
                    )
                else:
                    # PTPC-FP8 needs no per-K scaling: dedicated no-scale E4M3 WMMA.
                    accs[idx] = rocdl.wmma_f32_16x16x128_fp8_fp8(
                        T.vec(8, T.f32), b_frag, a_frag, accs[idx]
                    )
                return
            if const_expr(use_ascale_vgpr and ascale_opsel):
                # VGPR path pairs M-blocks across the two lane_kgrp halves.
                a_scale_idx = wm % ascale_half
                a_opsel = wm // ascale_half
            elif const_expr(use_ascale_shuffled_tdm and as32_opsel):
                # Shuffled path pairs adjacent 16-M WMMAs in one 32-row block.
                a_scale_idx = wm // 2
                a_opsel = wm % 2
            else:
                a_scale_idx = wm
                a_opsel = 0

            if const_expr(is_fp4):
                # 32x16 WMMA with A/B swap: SRC0=B, SRC1=A. 32x4 reads one 32-N block
                # per WMMA (idx wn).
                accs[idx] = rocdl.wmma_scale_f32_32x16x128_f4(
                    T.vec(16, T.f32),
                    b_frag,
                    a_frag,
                    accs[idx],
                    b_scales[wn],
                    a_scales[a_scale_idx],
                    scaleAType=0,
                    scaleBType=a_opsel,
                )
            else:
                # 16x16x128 WMMA: A8W4 (fmtA=FP4) or FP8 (fmtA=FP8). op_sel pairs
                # adjacent 16-N halves (32x4 even rep); else one scale per WMMA
                # (32x4 odd rep, or no op_sel).
                if const_expr(bs32_opsel):
                    b_scale_idx = wn // 2
                    b_opsel = wn % 2
                else:
                    b_scale_idx = wn
                    b_opsel = 0
                accs[idx] = rocdl.wmma_scale_f32_16x16x128_f8f6f4(
                    T.vec(8, T.f32),
                    b_frag,
                    a_frag,
                    accs[idx],
                    b_scales[b_scale_idx],
                    a_scales[a_scale_idx],
                    fmtA=4 if is_a8w4 else 0,
                    fmtB=0,
                    scaleAType=b_opsel,
                    scaleBType=a_opsel,
                )

        def _a_streaming_compute(
            accs,
            a_buf,
            a_bases,
            b_frags,
            b_scales,
            a_scales,
            ks,
            emit_filler=None,
            next_bs_info=None,
            mid_compute_callback=None,
        ):
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
                    is_last = wm == wmma_m_rep - 1
                    if const_expr(is_last and emit_filler is not None):
                        rocdl.sched_barrier(0)
                        emit_filler()
                    for wn_raw in range_constexpr(wmma_n_rep):
                        wn = (wmma_n_rep - 1 - wn_raw) if (wm % 2 == 1) else wn_raw
                        _emit_wmma(
                            accs,
                            wm,
                            wn,
                            a_frags[frag_i],
                            b_frags[wn],
                            a_scales,
                            b_scales,
                        )

            a_frags_front = [
                load_a_frag(a_buf, a_bases[wm], ks) for wm in range_constexpr(_front_wm)
            ]

            _use_partial_drain = (
                next_bs_info is not None and _front_wm * wmma_n_rep >= 4
            )

            if const_expr(_use_partial_drain):
                nb_buf, nb_bases, nas_buf, nas_bases, nbs_buf, nbs_bases, n_ks = (
                    next_bs_info
                )
                next_result = _load_b_and_scales(
                    nb_buf, nb_bases, nas_buf, nas_bases, nbs_buf, nbs_bases, n_ks
                )
                rocdl.s_wait_dscnt(_bs_ds_loads)
            else:
                rocdl.s_wait_dscnt(0)

            _emit_rows(0, a_frags_front)

            if const_expr(mid_compute_callback is not None):
                rocdl.sched_barrier(0)
                mid_compute_callback()

            if const_expr(_back_wm > 0):
                a_frags_back = [
                    load_a_frag(a_buf, a_bases[_front_wm + h], ks)
                    for h in range_constexpr(_back_wm)
                ]
                _back_drain = _bs_ds_loads if _use_partial_drain else 0
                rocdl.s_wait_dscnt(_back_drain)
                _emit_rows(_front_wm, a_frags_back)

            if const_expr(_use_partial_drain):
                return accs, next_result
            if const_expr(next_bs_info is not None):
                nb_buf, nb_bases, nas_buf, nas_bases, nbs_buf, nbs_bases, n_ks = (
                    next_bs_info
                )
                next_result = _load_b_and_scales(
                    nb_buf, nb_bases, nas_buf, nas_bases, nbs_buf, nbs_bases, n_ks
                )
                return accs, next_result
            return accs

        # ── Compute on one LDS buffer ──
        def compute_tile(
            accs_in,
            lds_a,
            lds_b,
            lds_as,
            lds_bs,
            emit_filler=None,
            mid_compute_callback=None,
            late_compute_callback=None,
            scale_k_base=None,
            pf_a_scales=None,
        ):
            current_accs = list(accs_in)
            _set_vgpr_a_scales(scale_k_base, pf_a_scales)
            a_buf, a_bases = _precompute_a_lane_bases(lds_a)
            b_buf, b_bases = _precompute_b_lane_bases(lds_b)
            if const_expr(is_mxscale):
                as_buf, as_bases = _precompute_as32_bases(lds_as)
                bs_buf, bs_bases = _precompute_bs32_bases(lds_bs)
            else:
                as_buf, as_bases = lds_as, None
                bs_buf, bs_bases = (
                    lds_bs,
                    None,
                )  # ptpc: B-scale in epilogue, bases unused

            if const_expr(k_wmma_steps == 1):
                b_frags, b_scales, a_scales = _load_b_and_scales(
                    b_buf, b_bases, as_buf, as_bases, bs_buf, bs_bases, 0
                )
                current_accs = _a_streaming_compute(
                    current_accs,
                    a_buf,
                    a_bases,
                    b_frags,
                    b_scales,
                    a_scales,
                    0,
                    emit_filler=emit_filler,
                    mid_compute_callback=mid_compute_callback,
                )
            else:
                if const_expr(use_row_major_k_prefetch):

                    def _load_bundle(ks):
                        b_frags, b_scales, a_scales = _load_b_and_scales(
                            b_buf, b_bases, as_buf, as_bases, bs_buf, bs_bases, ks
                        )
                        a_frag = load_a_frag(a_buf, a_bases[0], ks)
                        return a_frag, b_frags, a_scales, b_scales

                    def _emit_bundle(bundle, emit_filler_now=False):
                        a_frag, b_frags, a_scales, b_scales = bundle
                        if const_expr(emit_filler_now and emit_filler is not None):
                            rocdl.sched_barrier(0)
                            emit_filler()
                        for wn in range_constexpr(wmma_n_rep):
                            _emit_wmma(
                                current_accs,
                                0,
                                wn,
                                a_frag,
                                b_frags[wn],
                                a_scales,
                                b_scales,
                            )

                    # Keep future K-subtile LDS reads outstanding while only draining
                    # the current bundle before its single row-major WMMA.
                    preload_depth = min(k_wmma_steps, _row_major_k_prefetch_depth + 1)
                    bundle_queue = [
                        _load_bundle(pre_ks)
                        for pre_ks in range_constexpr(preload_depth)
                    ]
                    next_ks = preload_depth
                    for ks in range_constexpr(k_wmma_steps):
                        is_last_ks = ks == k_wmma_steps - 1
                        cur_bundle = bundle_queue.pop(0)
                        rocdl.s_wait_dscnt(
                            len(bundle_queue) * _row_major_k_prefetch_bundle_ds
                        )

                        if const_expr(is_last_ks and late_compute_callback is not None):
                            rocdl.sched_barrier(0)
                            late_compute_callback()

                        _emit_bundle(cur_bundle, emit_filler_now=is_last_ks)

                        if const_expr(ks == 0 and mid_compute_callback is not None):
                            rocdl.sched_barrier(0)
                            mid_compute_callback()

                        if const_expr(next_ks < k_wmma_steps):
                            bundle_queue.append(_load_bundle(next_ks))
                            next_ks += 1

                    return current_accs

                prev_b, prev_bs, prev_as = _load_b_and_scales(
                    b_buf, b_bases, as_buf, as_bases, bs_buf, bs_bases, 0
                )
                for ks in range_constexpr(k_wmma_steps - 1):
                    _mid_cb = mid_compute_callback if ks == 0 else None
                    current_accs, (prev_b, prev_bs, prev_as) = _a_streaming_compute(
                        current_accs,
                        a_buf,
                        a_bases,
                        prev_b,
                        prev_bs,
                        prev_as,
                        ks,
                        next_bs_info=(
                            b_buf,
                            b_bases,
                            as_buf,
                            as_bases,
                            bs_buf,
                            bs_bases,
                            ks + 1,
                        ),
                        mid_compute_callback=_mid_cb,
                    )
                current_accs = _a_streaming_compute(
                    current_accs,
                    a_buf,
                    a_bases,
                    prev_b,
                    prev_bs,
                    prev_as,
                    k_wmma_steps - 1,
                    emit_filler=emit_filler,
                )
            return current_accs

        def compute_tile_fp4_quadrant(
            accs_in,
            lds_a,
            lds_b,
            lds_as,
            lds_bs,
            emit_filler=None,
            mid_compute_callback=None,
            scale_k_base=None,
            pf_a_scales=None,
        ):
            current_accs = list(accs_in)
            _set_vgpr_a_scales(scale_k_base, pf_a_scales)
            a_buf, a_bases = _precompute_a_lane_bases(lds_a)
            b_buf, b_bases = _precompute_b_lane_bases(lds_b)
            as_buf, as_bases = _precompute_as32_bases(lds_as)
            bs_buf, bs_bases = _precompute_bs32_bases(lds_bs)
            _b_half_scale_loads = _fp4_half_wn  # 32x4: one b32 per 32-N block/WMMA

            def _fp4_get_a_scale_and_opsel(a_scales_all, wm_idx):
                if const_expr(use_ascale_vgpr and ascale_opsel):
                    return a_scales_all[wm_idx % ascale_half], wm_idx // ascale_half
                if const_expr(use_ascale_shuffled_tdm and as32_opsel):
                    return a_scales_all[wm_idx // 2], wm_idx % 2
                return a_scales_all[wm_idx], 0

            def _load_a_group(wm_base, wm_count, ks):
                return [
                    load_a_frag(a_buf, a_bases[wm_base + wm_local], ks)
                    for wm_local in range_constexpr(wm_count)
                ]

            def _load_b_half(wn_base, ks):
                return [
                    load_b_frag(b_buf, b_bases, wn_base + wn_local, ks)
                    for wn_local in range_constexpr(_fp4_half_wn)
                ]

            def _load_bs32_b_half(block0, wn_base, ks):
                # 32x4: load this N-half's blocks, one ds_load_b32 per 32-N WMMA (no op_sel).
                return _load_scale32_full_blocks(
                    bs_buf,
                    block0 + arith.index(wn_base),
                    ks,
                    bs32_lds_row_stride,
                    bs32_block_bytes,
                    _fp4_half_wn,
                )

            def _load_b_half_bundle(wn_base, ks):
                b_frags = _load_b_half(wn_base, ks)
                b_scales = _load_bs32_b_half(bs_bases, wn_base, ks)
                return b_frags, b_scales

            def _emit_group_rows(
                wn_base,
                wm_base,
                a_frags,
                b_frags,
                a_scales,
                b_scales,
                row_start,
                row_count,
                emit_filler_now=False,
            ):
                if const_expr(emit_filler_now and emit_filler is not None):
                    rocdl.sched_barrier(0)
                    emit_filler()
                for row_offset in range_constexpr(row_count):
                    wm_local = row_start + row_offset
                    a_frag = a_frags[wm_local]
                    global_wm = wm_base + wm_local
                    a_scale, a_opsel = _fp4_get_a_scale_and_opsel(a_scales, global_wm)
                    for wn_local in range_constexpr(_fp4_half_wn):
                        idx = global_wm * wmma_n_rep + (
                            wn_base + wn_local
                        )  # row-major slot
                        current_accs[idx] = rocdl.wmma_scale_f32_32x16x128_f4(
                            T.vec(16, T.f32),
                            b_frags[wn_local],
                            a_frag,
                            current_accs[idx],
                            b_scales[wn_local],
                            a_scale,
                            scaleAType=0,
                            scaleBType=a_opsel,
                        )

            def _emit_group(
                wn_base,
                wm_base,
                a_frags,
                b_frags,
                a_scales,
                b_scales,
                emit_filler_now=False,
            ):
                _emit_group_rows(
                    wn_base,
                    wm_base,
                    a_frags,
                    b_frags,
                    a_scales,
                    b_scales,
                    0,
                    _fp4_half_wm,
                    emit_filler_now=emit_filler_now,
                )

            b_left_frags, b_left_scales = _load_b_half_bundle(0, 0)

            for ks in range_constexpr(k_wmma_steps):
                is_last_ks = ks == k_wmma_steps - 1
                a_scales_all = _load_a_scale_operand(as_buf, as_bases, ks)

                a_top_frags = _load_a_group(0, _fp4_half_wm, ks)
                a_bottom_frags = _load_a_group(_fp4_half_wm, _fp4_half_wm, ks)

                # Wait for bottom-A loads; top-A stays in flight during Q1.
                rocdl.s_wait_dscnt(_fp4_half_wm * DS_LOADS_PER_A_FRAG)

                _emit_group(
                    0,
                    0,
                    a_top_frags,
                    b_left_frags,
                    a_scales_all,
                    b_left_scales,
                )

                if const_expr(ks == 0 and mid_compute_callback is not None):
                    rocdl.sched_barrier(0)
                    mid_compute_callback()

                b_right_frags, b_right_scales = _load_b_half_bundle(_fp4_half_wn, ks)

                # Hold only the next B half outstanding while the second
                # quadrant consumes the current left-half fragments.
                rocdl.s_wait_dscnt(_fp4_half_wn * 4 + _b_half_scale_loads)

                _emit_group(
                    0,
                    _fp4_half_wm,
                    a_bottom_frags,
                    b_left_frags,
                    a_scales_all,
                    b_left_scales,
                )

                if const_expr(not is_last_ks):
                    next_left_frags, next_left_scales = _load_b_half_bundle(0, ks + 1)
                    # Older right-half loads must be ready before consuming
                    # them, while the next ks left-half preload can remain in
                    # flight under the final two quadrants.
                    rocdl.s_wait_dscnt(_fp4_half_wn * 4 + _b_half_scale_loads)
                else:
                    rocdl.s_wait_dscnt(0)

                _emit_group(
                    _fp4_half_wn,
                    0,
                    a_top_frags,
                    b_right_frags,
                    a_scales_all,
                    b_right_scales,
                )
                _emit_group(
                    _fp4_half_wn,
                    _fp4_half_wm,
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

        def compute_tile_fp8_quadrant(
            accs_in,
            lds_a,
            lds_b,
            lds_as,
            lds_bs,
            emit_filler=None,
            mid_compute_callback=None,
            late_compute_callback=None,
            scale_k_base=None,
            pf_a_scales=None,
        ):
            current_accs = list(accs_in)
            _set_vgpr_a_scales(scale_k_base, pf_a_scales)
            a_buf, a_bases = _precompute_a_lane_bases(lds_a)
            b_buf, b_bases = _precompute_b_lane_bases(lds_b)
            if const_expr(is_mxscale):
                as_buf, as_bases = _precompute_as32_bases(lds_as)
                bs_buf, bs_bases = _precompute_bs32_bases(lds_bs)
            else:
                as_buf, as_bases = lds_as, None
                bs_buf, bs_bases = (
                    lds_bs,
                    None,
                )  # ptpc: B-scale in epilogue, bases unused
            _b_half_loads = _fp8_half_wn * _b_frag_loads_per_wn
            _b_left_bundle_loads = _b_half_loads + _fp8_b_scale_loads

            def _load_a_group(wm_base, wm_count, ks):
                return [
                    load_a_frag(a_buf, a_bases[wm_base + wm_local], ks)
                    for wm_local in range_constexpr(wm_count)
                ]

            def _load_b_half(wn_base, ks):
                return [
                    load_b_frag(b_buf, b_bases, wn_base + wn_local, ks)
                    for wn_local in range_constexpr(_fp8_half_wn)
                ]

            def _load_a_scales(ks):
                if const_expr(is_ptpc):
                    return None  # PTPC: scale applied in epilogue, not in K-loop
                return _load_a_scale_operand(as_buf, as_bases, ks)

            def _load_b_scales(ks):
                if const_expr(is_ptpc):
                    return None  # PTPC: scale applied in epilogue, not in K-loop
                return _load_b_scale_lds(
                    bs_buf, bs_bases, ks
                )  # 32x4; op_sel in _emit_wmma

            def _load_b_left_bundle(ks):
                return _load_b_half(0, ks), _load_b_scales(ks)

            def _emit_group_rows(
                wm_base,
                wn_base,
                a_frags,
                b_frags,
                a_scales,
                b_scales,
                row_start,
                row_count,
                emit_filler_now=False,
            ):
                if const_expr(emit_filler_now and emit_filler is not None):
                    rocdl.sched_barrier(0)
                    emit_filler()
                for row_offset in range_constexpr(row_count):
                    wm_local = row_start + row_offset
                    global_wm = wm_base + wm_local
                    for wn_local in range_constexpr(_fp8_half_wn):
                        global_wn = wn_base + wn_local
                        _emit_wmma(
                            current_accs,
                            global_wm,
                            global_wn,
                            a_frags[wm_local],
                            b_frags[wn_local],
                            a_scales,
                            b_scales,
                        )

            def _emit_group(
                wm_base,
                wn_base,
                a_frags,
                b_frags,
                a_scales,
                b_scales,
                emit_filler_now=False,
            ):
                _emit_group_rows(
                    wm_base,
                    wn_base,
                    a_frags,
                    b_frags,
                    a_scales,
                    b_scales,
                    0,
                    _fp8_half_wm,
                    emit_filler_now=emit_filler_now,
                )

            def _emit_group_col(
                wm_base, wn_base, a_frags, b_frags, a_scales, b_scales, wn_local
            ):
                global_wn = wn_base + wn_local
                for wm_local in range_constexpr(_fp8_half_wm):
                    global_wm = wm_base + wm_local
                    _emit_wmma(
                        current_accs,
                        global_wm,
                        global_wn,
                        a_frags[wm_local],
                        b_frags[wn_local],
                        a_scales,
                        b_scales,
                    )

            b_left_frags, b_scales = _load_b_left_bundle(0)
            # Margin = a-top drain depth (b-scale is issued earlier, so it is unrelated);
            # keep it at the per-WMMA count so op_sel's fewer b-scale loads don't widen
            # keep and race the top-row A frags.
            _top_keep_margin = (
                b_scale_load_rep if const_expr(bs32_opsel) else _fp8_b_scale_loads
            )
            _first_top_row_keep = max(
                (_fp8_half_wm - 1) * DS_LOADS_PER_A_FRAG - _top_keep_margin, 0
            )
            _bottom_left_keep = max(_b_half_loads - DS_LOADS_PER_A_FRAG, 0)

            for ks in range_constexpr(k_wmma_steps):
                is_last_ks = ks == k_wmma_steps - 1
                a_scales = _load_a_scales(ks)

                a_top_frags = _load_a_group(0, _fp8_half_wm, ks)

                # Consume the first top-left row before issuing bottom-A.
                # The barriers only constrain LLVM scheduling; they are not
                # hardware synchronization points.
                rocdl.s_wait_dscnt(_first_top_row_keep)
                rocdl.sched_barrier(0)
                _emit_group_rows(
                    0, 0, a_top_frags, b_left_frags, a_scales, b_scales, 0, 1
                )
                rocdl.sched_barrier(0)

                a_bottom_frags = _load_a_group(_fp8_half_wm, _fp8_half_wm, ks)
                if const_expr(_fp8_half_wm > 1):
                    _emit_group_rows(
                        0,
                        0,
                        a_top_frags,
                        b_left_frags,
                        a_scales,
                        b_scales,
                        1,
                        _fp8_half_wm - 1,
                    )
                b_right_frags = _load_b_half(_fp8_half_wn, ks)

                # Drain bottom-A while keeping most right-half B in flight.
                rocdl.s_wait_dscnt(_bottom_left_keep)

                _emit_group(
                    _fp8_half_wm, 0, a_bottom_frags, b_left_frags, a_scales, b_scales
                )

                if const_expr(ks == 0 and mid_compute_callback is not None):
                    rocdl.sched_barrier(0)
                    mid_compute_callback()

                if const_expr(not is_last_ks):
                    next_left_frags, next_b_scales = _load_b_left_bundle(ks + 1)

                for wn_local in range_constexpr(_fp8_half_wn):
                    if const_expr(not is_last_ks):
                        _right_keep = (
                            _b_left_bundle_loads
                            + (_fp8_half_wn - wn_local - 1) * _b_frag_loads_per_wn
                        )
                    else:
                        _right_keep = (
                            _fp8_half_wn - wn_local - 1
                        ) * _b_frag_loads_per_wn
                    rocdl.s_wait_dscnt(_right_keep)
                    _emit_group_col(
                        0,
                        _fp8_half_wn,
                        a_top_frags,
                        b_right_frags,
                        a_scales,
                        b_scales,
                        wn_local,
                    )

                if const_expr(is_last_ks and late_compute_callback is not None):
                    rocdl.sched_barrier(0)
                    late_compute_callback()

                if const_expr(is_last_ks and emit_filler is not None):
                    rocdl.sched_barrier(0)
                    emit_filler()

                for wn_local in range_constexpr(_fp8_half_wn):
                    _emit_group_col(
                        _fp8_half_wm,
                        _fp8_half_wn,
                        a_bottom_frags,
                        b_right_frags,
                        a_scales,
                        b_scales,
                        wn_local,
                    )

                if const_expr(not is_last_ks):
                    b_left_frags = next_left_frags
                    b_scales = next_b_scales

            return current_accs

        def compute_tile_fp8_deep_pipeline(
            accs_in,
            lds_a,
            lds_b,
            lds_as,
            lds_bs,
            emit_filler=None,
            mid_compute_callback=None,
            late_compute_callback=None,
            a0_prefetch=None,
            scale_k_base=None,
            pf_a_scales=None,
        ):
            current_accs = list(accs_in)
            _set_vgpr_a_scales(scale_k_base, pf_a_scales)
            a_buf, a_bases = _precompute_a_lane_bases(lds_a)
            b_buf, b_bases = _precompute_b_lane_bases(lds_b)
            if const_expr(is_mxscale):
                as_buf, as_bases = _precompute_as32_bases(lds_as)
                bs_buf, bs_bases = _precompute_bs32_bases(lds_bs)
            else:
                as_buf, as_bases = lds_as, None
                bs_buf, bs_bases = (
                    lds_bs,
                    None,
                )  # ptpc: B-scale in epilogue, bases unused

            def load_a_pair(wm_pair, ks):
                wm_base = wm_pair * _fp8_pair_wm
                return [
                    load_a_frag(a_buf, a_bases[wm_base + wm_local], ks)
                    for wm_local in range_constexpr(_fp8_pair_wm)
                ]

            def load_b_pair(wn_pair, ks):
                wn_base = wn_pair * _fp8_pair_wn
                return [
                    load_b_frag(b_buf, b_bases, wn_base + wn_local, ks)
                    for wn_local in range_constexpr(_fp8_pair_wn)
                ]

            def emit_panel_2x2(
                wm_pair,
                wn_pair,
                a_pair,
                b_pair,
                scale_pair,
                prefetch_after_first_row=None,
            ):
                a_scales, b_scales = scale_pair
                wm_base = wm_pair * _fp8_pair_wm
                wn_base = wn_pair * _fp8_pair_wn
                for wn_local in range_constexpr(_fp8_pair_wn):
                    _emit_wmma(
                        current_accs,
                        wm_base,
                        wn_base + wn_local,
                        a_pair[0],
                        b_pair[wn_local],
                        a_scales,
                        b_scales,
                    )
                if const_expr(prefetch_after_first_row is not None):
                    prefetch_after_first_row()
                for wn_local in range_constexpr(_fp8_pair_wn):
                    _emit_wmma(
                        current_accs,
                        wm_base + 1,
                        wn_base + wn_local,
                        a_pair[1],
                        b_pair[wn_local],
                        a_scales,
                        b_scales,
                    )

            def emit_panel_2x2_row(
                wm_pair, wn_pair, row_local, a_pair, b_pair, scale_pair
            ):
                a_scales, b_scales = scale_pair
                wm_base = wm_pair * _fp8_pair_wm
                wn_base = wn_pair * _fp8_pair_wn
                for wn_local in range_constexpr(_fp8_pair_wn):
                    _emit_wmma(
                        current_accs,
                        wm_base + row_local,
                        wn_base + wn_local,
                        a_pair[row_local],
                        b_pair[wn_local],
                        a_scales,
                        b_scales,
                    )

            _pair_loads = _fp8_pair_a_loads
            _two_pair_loads = _fp8_pair_a_loads + _fp8_pair_b_loads

            for ks in range_constexpr(k_wmma_steps):
                is_last_ks = ks == k_wmma_steps - 1
                a_scales, b_scales = _scales_for_emit(
                    as_buf, as_bases, bs_buf, bs_bases, ks
                )
                scale_pair = (a_scales, b_scales)

                b0 = load_b_pair(0, ks)
                if const_expr(
                    ks == 0
                    and a0_prefetch is not None
                    and len(a0_prefetch) == _fp8_pair_wm
                ):
                    a0 = list(a0_prefetch)
                elif const_expr(ks == 0 and a0_prefetch is not None):
                    a0 = [a0_prefetch[0], load_a_frag(a_buf, a_bases[1], ks)]
                else:
                    a0 = load_a_pair(0, ks)
                b1 = load_b_pair(1, ks)
                b2 = load_b_pair(2, ks)

                a1_box = [None]
                b3_box = [None]
                a2_box = [None]
                a3_box = [None]

                def _prefetch_a1():
                    a1_box[0] = load_a_pair(1, ks)

                first_wait_keep = _two_pair_loads + 3
                if const_expr(ks == 0 and a0_prefetch is not None):
                    first_wait_keep += DS_LOADS_PER_A_FRAG * len(a0_prefetch)
                rocdl.s_wait_dscnt(first_wait_keep)
                emit_panel_2x2(
                    0, 0, a0, b0, scale_pair, prefetch_after_first_row=_prefetch_a1
                )

                if const_expr(ks == 0 and mid_compute_callback is not None):
                    rocdl.sched_barrier(0)
                    mid_compute_callback()

                def _prefetch_b3():
                    b3_box[0] = load_b_pair(3, ks)

                def _prefetch_a3():
                    a3_box[0] = load_a_pair(3, ks)

                rocdl.s_wait_dscnt(_pair_loads + _fp8_pair_b_loads)
                emit_panel_2x2(
                    0, 1, a0, b1, scale_pair, prefetch_after_first_row=_prefetch_b3
                )

                rocdl.s_wait_dscnt(_fp8_pair_b_loads + 2)
                emit_panel_2x2(
                    1,
                    0,
                    a1_box[0],
                    b0,
                    scale_pair,
                    prefetch_after_first_row=_prefetch_a3,
                )

                def _prefetch_a2():
                    a2_box[0] = load_a_pair(2, ks)

                emit_panel_2x2(1, 1, a1_box[0], b1, scale_pair)

                emit_panel_2x2(
                    0, 2, a0, b2, scale_pair, prefetch_after_first_row=_prefetch_a2
                )
                emit_panel_2x2_row(1, 2, 0, a1_box[0], b2, scale_pair)
                emit_panel_2x2_row(1, 2, 1, a1_box[0], b2, scale_pair)
                rocdl.s_wait_dscnt(_pair_loads)
                emit_panel_2x2(0, 3, a0, b3_box[0], scale_pair)
                emit_panel_2x2(1, 3, a1_box[0], b3_box[0], scale_pair)

                emit_panel_2x2(2, 0, a2_box[0], b0, scale_pair)
                if const_expr(is_last_ks and late_compute_callback is not None):
                    rocdl.sched_barrier(0)
                    late_compute_callback()
                emit_panel_2x2(2, 1, a2_box[0], b1, scale_pair)

                rocdl.s_wait_dscnt(0)
                emit_panel_2x2(3, 0, a3_box[0], b0, scale_pair)
                emit_panel_2x2(3, 1, a3_box[0], b1, scale_pair)

                if const_expr(is_last_ks and emit_filler is not None):
                    rocdl.sched_barrier(0)
                    emit_filler()

                emit_panel_2x2(2, 2, a2_box[0], b2, scale_pair)
                emit_panel_2x2(2, 3, a2_box[0], b3_box[0], scale_pair)
                emit_panel_2x2(3, 2, a3_box[0], b2, scale_pair)
                emit_panel_2x2(3, 3, a3_box[0], b3_box[0], scale_pair)

            return current_accs

        def hot_loop_scheduler():
            if const_expr(use_row_major_k_prefetch):
                _queue_depth = min(k_wmma_steps, _row_major_k_prefetch_depth + 1)
                for _ks in range_constexpr(k_wmma_steps):
                    if const_expr(_ks == 0):
                        rocdl.sched_dsrd(_row_major_k_prefetch_bundle_ds * _queue_depth)
                    elif const_expr(_ks + _queue_depth <= k_wmma_steps):
                        rocdl.sched_dsrd(_row_major_k_prefetch_bundle_ds)
                    rocdl.sched_mfma(wmma_n_rep)
                rocdl.sched_barrier(0)
                return

            _half_wm = wmma_m_rep // 2
            _half_wmma = _half_wm * wmma_n_rep
            _b_loads_per_frag = 2 if is_a8w4 else 4
            _scale_dsrd = _scale_ds_loads
            _a_half_dsrd = _half_wm * DS_LOADS_PER_A_FRAG

            for _ks in range_constexpr(k_wmma_steps):
                if const_expr(_ks == 0):
                    rocdl.sched_dsrd(
                        wmma_n_rep * _b_loads_per_frag + _scale_dsrd + _a_half_dsrd
                    )
                else:
                    rocdl.sched_dsrd(_a_half_dsrd)
                rocdl.sched_mfma(_half_wmma)
                rocdl.sched_dsrd(_a_half_dsrd)
                rocdl.sched_mfma(_half_wmma)
                if const_expr(_ks < k_wmma_steps - 1):
                    rocdl.sched_dsrd(wmma_n_rep * _b_loads_per_frag + _scale_dsrd)
            rocdl.sched_barrier(0)

        def hot_loop_scheduler_fp4_quadrant():
            _a_all_loads = wmma_m_rep * DS_LOADS_PER_A_FRAG
            _a_scale_loads = _a_scale_ds
            _b_half_loads = _fp4_half_wn * 4
            _b_half_scale_loads = _fp4_half_wn  # 32x4: one b32 per 32-N block/WMMA
            _group_wmma = _fp4_group_size
            _right_half_loads = _b_half_loads + _b_half_scale_loads

            for _ks in range_constexpr(k_wmma_steps):
                if const_expr(_ks == 0):
                    rocdl.sched_dsrd(
                        _a_all_loads
                        + _a_scale_loads
                        + _b_half_loads
                        + _b_half_scale_loads
                    )
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

        def hot_loop_scheduler_fp8_quadrant():
            _a_scale_loads = _a_scale_ds
            _a_top_loads = _fp8_half_wm * DS_LOADS_PER_A_FRAG
            _a_bottom_loads = _a_top_loads
            _b_half_loads = _fp8_half_wn * _b_frag_loads_per_wn
            _b_left_bundle_loads = _b_half_loads + _fp8_b_scale_loads
            _group_wmma = _fp8_group_size
            _first_row_wmma = _fp8_half_wn
            _remaining_top_left_wmma = (_fp8_half_wm - 1) * _fp8_half_wn

            for _ks in range_constexpr(k_wmma_steps):
                if const_expr(_ks == 0):
                    rocdl.sched_dsrd(
                        _b_left_bundle_loads + _a_scale_loads + _a_top_loads
                    )
                else:
                    rocdl.sched_dsrd(_a_scale_loads + _a_top_loads)
                rocdl.sched_mfma(_first_row_wmma)
                rocdl.sched_dsrd(_a_bottom_loads)
                if const_expr(_remaining_top_left_wmma > 0):
                    rocdl.sched_mfma(_remaining_top_left_wmma)
                rocdl.sched_dsrd(_b_half_loads)
                rocdl.sched_mfma(_group_wmma)
                if const_expr(_ks < k_wmma_steps - 1):
                    rocdl.sched_dsrd(_b_left_bundle_loads)
                for _wn_local in range_constexpr(_fp8_half_wn):
                    rocdl.sched_mfma(_fp8_half_wm)
                for _wn_local in range_constexpr(_fp8_half_wn):
                    rocdl.sched_mfma(_fp8_half_wm)
            rocdl.sched_barrier(0)

        def hot_loop_scheduler_fp8_deep_pipeline():
            def _sched_panel_2x2(prefetch_loads=0):
                if const_expr(prefetch_loads > 0):
                    rocdl.sched_mfma(_fp8_pair_wn)
                    rocdl.sched_dsrd(prefetch_loads)
                    rocdl.sched_mfma(_fp8_pair_wn)
                else:
                    rocdl.sched_mfma(_fp8_pair_wm * _fp8_pair_wn)

            def _sched_panel_row():
                rocdl.sched_mfma(_fp8_pair_wn)

            _initial_loads = (
                _fp8_scale_loads + _fp8_pair_b_loads * 3 + _fp8_pair_a_loads
            )

            for _ks in range_constexpr(k_wmma_steps):
                _ks_initial_loads = _initial_loads
                if const_expr(_ks == 0):
                    _ks_initial_loads -= _fp8_pair_a_loads
                rocdl.sched_dsrd(_ks_initial_loads)
                _sched_panel_2x2(_fp8_pair_a_loads)
                _sched_panel_2x2(_fp8_pair_b_loads)
                _sched_panel_2x2(_fp8_pair_a_loads)
                _sched_panel_2x2()
                _sched_panel_2x2(_fp8_pair_a_loads)
                _sched_panel_row()
                _sched_panel_row()
                _sched_panel_2x2()
                _sched_panel_2x2()
                _sched_panel_2x2()
                _sched_panel_2x2()
                _sched_panel_2x2()
                _sched_panel_2x2()
                _sched_panel_2x2()
                _sched_panel_2x2()
                _sched_panel_2x2()
                _sched_panel_2x2()
            rocdl.sched_barrier(0)

        def compute_tile_scheduled(
            accs_in,
            lds_a,
            lds_b,
            lds_as,
            lds_bs,
            emit_filler=None,
            mid_compute_callback=None,
            late_compute_callback=None,
            a0_prefetch=None,
            scale_k_base=None,
            pf_a_scales=None,
        ):
            if const_expr(compute_schedule_kind == COMPUTE_SCHEDULE_FP4_QUADRANT):
                return compute_tile_fp4_quadrant(
                    accs_in,
                    lds_a,
                    lds_b,
                    lds_as,
                    lds_bs,
                    emit_filler=emit_filler,
                    mid_compute_callback=mid_compute_callback,
                    scale_k_base=scale_k_base,
                    pf_a_scales=pf_a_scales,
                )
            if const_expr(compute_schedule_kind == COMPUTE_SCHEDULE_FP8_QUADRANT):
                return compute_tile_fp8_quadrant(
                    accs_in,
                    lds_a,
                    lds_b,
                    lds_as,
                    lds_bs,
                    emit_filler=emit_filler,
                    mid_compute_callback=mid_compute_callback,
                    late_compute_callback=late_compute_callback,
                    scale_k_base=scale_k_base,
                    pf_a_scales=pf_a_scales,
                )
            if const_expr(compute_schedule_kind == COMPUTE_SCHEDULE_FP8_DEEP_PIPELINE):
                return compute_tile_fp8_deep_pipeline(
                    accs_in,
                    lds_a,
                    lds_b,
                    lds_as,
                    lds_bs,
                    emit_filler=emit_filler,
                    mid_compute_callback=mid_compute_callback,
                    late_compute_callback=late_compute_callback,
                    a0_prefetch=a0_prefetch,
                    scale_k_base=scale_k_base,
                    pf_a_scales=pf_a_scales,
                )
            return compute_tile(
                accs_in,
                lds_a,
                lds_b,
                lds_as,
                lds_bs,
                emit_filler=emit_filler,
                mid_compute_callback=mid_compute_callback,
                late_compute_callback=late_compute_callback,
                scale_k_base=scale_k_base,
                pf_a_scales=pf_a_scales,
            )

        def hot_loop_scheduler_scheduled():
            if const_expr(compute_schedule_kind == COMPUTE_SCHEDULE_FP4_QUADRANT):
                hot_loop_scheduler_fp4_quadrant()
            elif const_expr(
                compute_schedule_kind == COMPUTE_SCHEDULE_FP8_DEEP_PIPELINE
            ):
                hot_loop_scheduler_fp8_deep_pipeline()
            elif const_expr(compute_schedule_kind == COMPUTE_SCHEDULE_FP8_QUADRANT):
                hot_loop_scheduler_fp8_quadrant()
            else:
                hot_loop_scheduler()

        def prefetch_fp8_deep_a0_frags(lds_a):
            a_buf, a_bases = _precompute_a_lane_bases(lds_a)
            return [
                load_a_frag(a_buf, a_bases[wm_local], 0)
                for wm_local in range_constexpr(_fp8_pair_wm)
            ]

        def maybe_prefetch_fp8_deep_a0(lds_a):
            # Call only after the TDM fence for this stage; pre-fence LDS reads can race multicast delivery.
            if const_expr(use_fp8_deep_pipeline_schedule):
                return prefetch_fp8_deep_a0_frags(lds_a)
            return None

        # ── Epilogue (unified via _sub_tiles) ──
        def _get_acc_sub8(accs, acc_idx, vec_base):
            """Extract 8-element sub-vector from accumulator."""
            if const_expr(ACC_VEC_SIZE == 8):
                return accs[acc_idx]
            indices = [vec_base + i for i in range_constexpr(8)]
            acc = fx.Vector(accs[acc_idx])
            return acc.shuffle(acc, indices)

        def epilogue_prepare_addrs():
            addrs = []
            _bf16_out = out_dtype in ("bf16", "f16")
            for acc_idx, vec_base, m_off, wn in _sub_tiles:
                row = blk_m + warp_m_base + arith.index(m_off) + lane16
                col_base = (
                    blk_n
                    + warp_n_base
                    + arith.index(wn * WMMA_N)
                    + lane_kgrp * arith.index(8)
                )
                if const_expr(_bf16_out):
                    c_off_bytes = (row * n_stride + col_base) * arith.index(
                        elem_bytes_d
                    )
                    addrs.append(c_off_bytes)
                else:
                    for half in range_constexpr(2):
                        col = col_base + arith.index(half * 4)
                        c_off = row * n_stride + col
                        addrs.append(c_off)
            return addrs

        _bf16_out = out_dtype in ("bf16", "f16")
        _out_elem_local = (
            T.bf16 if out_dtype == "bf16" else (T.f16 if out_dtype == "f16" else None)
        )

        def epilogue_stores(final_accs, addrs):
            addr_idx = 0
            for acc_idx, vec_base, m_off, wn in _sub_tiles:
                sub8 = _get_acc_sub8(final_accs, acc_idx, vec_base)
                if const_expr(_bf16_out):
                    addr_idx += store_acc_vec8_to_buffer(
                        sub8,
                        c_rsrc,
                        addrs[addr_idx],
                        out_elem=_out_elem_local,
                        offset_is_bytes=True,
                    )
                else:
                    addr_idx += store_acc_vec8_to_buffer(
                        sub8, c_rsrc, addrs[addr_idx : addr_idx + 2]
                    )

        def epilogue_lds_stores(final_accs, d_buf, d_base):
            for acc_idx, vec_base, m_off, wn in _sub_tiles:
                sub8 = _get_acc_sub8(final_accs, acc_idx, vec_base)
                imm = m_off * _lds_d_stride_elems + wn * _n_col_d_elems
                store_acc_vec8_to_lds(
                    d_buf, d_base, imm, sub8, out_elem=_out_elem_local
                )

        def _atomic_fadd_global(val, byte_off):
            # Device-scoped, relaxed atomic add into C at c_global_base_i64 + byte_off.
            addr_i64 = llvm.AddOp(
                c_global_base_i64,
                arith.index_cast(T.i64, byte_off),
                llvm.IntegerOverflowFlags(0),
            ).result
            ptr = llvm.IntToPtrOp(c_global_ptr_type, addr_i64).result
            llvm.AtomicRMWOp(
                llvm.AtomicBinOp.fadd,
                ptr,
                val.ir_value(),
                llvm.AtomicOrdering.monotonic,
                syncscope="agent",
                alignment=4,
            )

        def _atomic_add_acc_vec8_to_buffer(acc_vec8, addr):
            if const_expr(_bf16_out):
                h_vec = fx.Vector(arith.trunc_f(T.vec(8, _out_elem_local), acc_vec8))
                for pair in range_constexpr(4):
                    pair_vec = fx.Vector.from_elements(
                        [h_vec[pair * 2], h_vec[pair * 2 + 1]]
                    )
                    byte_off = addr + arith.index(pair * 4)
                    _atomic_fadd_global(pair_vec, byte_off)
                return 1

            acc_vec = fx.Vector(acc_vec8)
            for half in range_constexpr(2):
                base_addr = addr[half] if isinstance(addr, (list, tuple)) else addr
                for vi in range_constexpr(4):
                    val = acc_vec[half * 4 + vi]
                    byte_off = (base_addr + arith.index(vi)) * arith.index(4)
                    _atomic_fadd_global(val, byte_off)
            return 2

        def epilogue_atomic_adds(final_accs, addrs):
            addr_idx = 0
            for acc_idx, vec_base, m_off, wn in _sub_tiles:
                sub8 = _get_acc_sub8(final_accs, acc_idx, vec_base)
                n_slots = 1 if _bf16_out else 2
                addr_arg = (
                    addrs[addr_idx] if _bf16_out else addrs[addr_idx : addr_idx + 2]
                )
                # Atomics use a raw global ptr (no num_records clip), so predicate
                # per-lane to skip rows >= M.
                row = blk_m + warp_m_base + arith.index(m_off) + lane16
                if_op = scf.IfOp(row < m_idx, [], has_else=False)
                with ir.InsertionPoint(if_op.then_block):
                    _atomic_add_acc_vec8_to_buffer(sub8, addr_arg)
                    scf.YieldOp([])
                addr_idx += n_slots

        def epilogue_load_ptpc_scales():
            # PTPC scales: sa[M] per-token (scalar per wm), sb[N] per-channel
            # (8 contiguous N cols per wn). Both fp32, constant along K.
            # The scale memrefs are dynamically shaped, so max_size=False would fall
            # back to a max-sized descriptor and disable hardware OOB. Derive
            # num_records from runtime M / compile-time N (fp32 = 4 bytes) so the
            # partial last M-tile clips rows >= M (and cols >= N) to 0.
            sa_rsrc = buffer_ops.create_buffer_resource(
                arg_a_scale, num_records_bytes=m_idx * arith.index(4)
            )
            sb_rsrc = buffer_ops.create_buffer_resource(
                arg_b_scale, num_records_bytes=N * 4
            )
            sa = []
            for wm in range_constexpr(wmma_m_rep):
                row = blk_m + warp_m_base + arith.index(wm * WMMA_M) + lane16
                sv = buffer_ops.buffer_load(
                    sa_rsrc, arith.index_cast(T.i32, row), vec_width=1, dtype=T.f32
                )
                sa.append(fx.Vector.from_elements([sv] * 8))
            sb = []
            for wn in range_constexpr(wmma_n_rep):
                col_base = (
                    blk_n
                    + warp_n_base
                    + arith.index(wn * WMMA_N)
                    + lane_kgrp * arith.index(8)
                )
                # buffer_load vec_width is capped at 4: read 8 cols as 2x vec4.
                lo = fx.Vector(
                    buffer_ops.buffer_load(
                        sb_rsrc,
                        arith.index_cast(T.i32, col_base),
                        vec_width=4,
                        dtype=T.f32,
                    )
                )
                hi = fx.Vector(
                    buffer_ops.buffer_load(
                        sb_rsrc,
                        arith.index_cast(T.i32, col_base + arith.index(4)),
                        vec_width=4,
                        dtype=T.f32,
                    )
                )
                sb.append(
                    fx.Vector.from_elements(
                        [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]]
                    )
                )
            return sa, sb

        def epilogue_apply_ptpc_scale(accs_in, sa, sb):
            out = list(accs_in)
            for wm in range_constexpr(wmma_m_rep):
                for wn in range_constexpr(wmma_n_rep):
                    idx = wm * wmma_n_rep + wn
                    out[idx] = (fx.Vector(out[idx]) * sb[wn] * sa[wm]).ir_value()
            return out

        _effective_l2_pf = l2_prefetch_distance
        if const_expr(use_cluster and l2_prefetch_distance > 0):
            _effective_l2_pf = max(1, l2_prefetch_distance - 1)

        def _l2_prefetch(k_base):
            if const_expr(_effective_l2_pf <= 0):
                return
            pf_k = k_base + arith.index(_effective_l2_pf * tile_k)
            pf_k_packed_a = pf_k // arith.index(PACK_FACTOR_A)
            pf_k_packed_b = pf_k // arith.index(PACK_FACTOR_B)
            tdm_ops.l2_prefetch_tile(
                arg_a,
                (blk_m, pf_k_packed_a),
                (tile_m, packed_tile_k_a),
                (K_packed_a, 1),
                elem_bytes=1,
                thread_id=tx,
                block_threads=block_threads,
            )
            tdm_ops.l2_prefetch_tile(
                arg_b,
                (blk_n // arith.index(16), pf_k_packed_b * arith.index(16)),
                (tile_n // 16, packed_tile_k_b * 16),
                (K_packed_b * 16, 1),
                elem_bytes=1,
                thread_id=tx,
                block_threads=block_threads,
            )

        # ====== Multi-stage pipeline ======
        acc_zero = arith.constant_vector(0.0, T.vec(ACC_VEC_SIZE, T.f32))
        accs = [acc_zero] * n_accs

        lds_a_data_f16 = lds_a_data_bytes // 2
        lds_b_data_f16 = lds_b_data_bytes // 2
        lds_a_scale_f16 = lds_a_scale_bytes // 2
        lds_b_scale_f16 = lds_b_scale_bytes // 2

        arena_base_ptr = arena_alloc.get_base()

        stages_a = [
            SmemPtr(
                arena_base_ptr,
                stage_a_data_off[i],
                elem_ty_lds,
                shape=(lds_a_data_f16,),
            )
            for i in range_constexpr(num_buffers)
        ]
        stages_b = [
            SmemPtr(
                arena_base_ptr,
                stage_b_data_off[i],
                elem_ty_lds,
                shape=(lds_b_data_f16,),
            )
            for i in range_constexpr(num_buffers)
        ]
        if const_expr(is_ptpc):
            # PTPC applies sa*sb in the epilogue from global memory: no scale LDS.
            # Alias the scale stage handles to A/B so the shared plumbing stays valid;
            # for PTPC they are never written (no scale TDM) or read.
            stages_as = stages_a
            stages_bs = stages_b
        else:
            stages_as = [
                SmemPtr(
                    arena_base_ptr,
                    stage_a_scale_off[i],
                    elem_ty_lds,
                    shape=(lds_a_scale_f16,),
                )
                for i in range_constexpr(num_buffers)
            ]
            stages_bs = [
                SmemPtr(
                    arena_base_ptr,
                    stage_b_scale_off[i],
                    elem_ty_lds,
                    shape=(lds_b_scale_f16,),
                )
                for i in range_constexpr(num_buffers)
            ]

        stages_a_mem = [stages_a[i].get() for i in range_constexpr(num_buffers)]
        stages_b_mem = [stages_b[i].get() for i in range_constexpr(num_buffers)]
        stages_as_mem = [stages_as[i].get() for i in range_constexpr(num_buffers)]
        stages_bs_mem = [stages_bs[i].get() for i in range_constexpr(num_buffers)]

        stages_a_idx = [
            extract_lds_base_idx(stages_a[i]) for i in range_constexpr(num_buffers)
        ]
        stages_b_idx = [
            extract_lds_base_idx(stages_b[i]) for i in range_constexpr(num_buffers)
        ]
        stages_as_idx = [
            extract_lds_base_idx(stages_as[i]) for i in range_constexpr(num_buffers)
        ]
        stages_bs_idx = [
            extract_lds_base_idx(stages_bs[i]) for i in range_constexpr(num_buffers)
        ]

        if const_expr(tdm_store_enabled):
            d_lds_base_ptr = arena_base_ptr
            d_lds_f16_count = total_d_bytes // 2
            d_smem = SmemPtr(
                d_lds_base_ptr, d_output_off, elem_ty_lds, shape=(d_lds_f16_count,)
            )
            d_lds_buffer = get_lds_memref(d_smem)
            warp_lds_off = (
                wave_m_idx * arith.index(n_warp) + wave_n_idx
            ) * arith.index(_warp_d_elems)
            d_lane_base = (
                warp_lds_off
                + lane16 * arith.index(_lds_d_stride_elems)
                + lane_kgrp * arith.index(4 * elem_bytes_d)
            )
            wave_id_idx = arith.index_cast(T.index, rocdl.wave_id())
            # Match the TDM-store descriptor offsets to the compute wave mapping.
            if const_expr(use_fp8_deep_pipeline_schedule):
                wave_m_sgpr = wave_id_idx % arith.index(m_warp)
                wave_n_sgpr = wave_id_idx // arith.index(m_warp)
            else:
                wave_m_sgpr = wave_id_idx // arith.index(n_warp)
                wave_n_sgpr = wave_id_idx % arith.index(n_warp)
            d_warp_linear_sgpr = wave_m_sgpr * arith.index(n_warp) + wave_n_sgpr
            d_warp_off_sgpr = d_warp_linear_sgpr * arith.index(
                warp_d_bytes
            ) + arith.index(d_output_off)
            warp_m_off_sgpr = wave_m_sgpr * arith.index(warp_tile_m)
            warp_n_off_sgpr = wave_n_sgpr * arith.index(warp_tile_n)
            d_desc = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_c,
                lds_memref=d_lds_base_ptr,
                global_offset=(blk_m + warp_m_off_sgpr, blk_n + warp_n_off_sgpr),
                tensor_shape=(warp_tile_m, warp_tile_n),
                strides=(n_stride, 1),
                tile_shape=(warp_tile_m, warp_tile_n),
                elem_bytes=elem_bytes_d,
                pad_interval=warp_tile_n,
                pad_amount=LDS_PAD_D_BYTES // elem_bytes_d,
                num_warps=1,
                lds_byte_offset=d_warp_off_sgpr,
                for_store=True,
                oob_outer_bound=i32_m,
            )

        # TDM descriptor lane layout: dgroup0 = [predicate, lds_addr, addr_lo, addr_hi].
        def _dg0_lane(desc, lane):
            return fx.Vector(desc.dgroup0)[lane]

        def _pack_dg0(pred, lds_addr, addr_lo, addr_hi):
            return fx.Vector.from_elements([pred, lds_addr, addr_lo, addr_hi], fx.Int32)

        # Precompute LDS addresses for TDM descriptor switching
        stages_a_lds_addr = []
        stages_b_lds_addr = []
        stages_as_lds_addr = []
        stages_bs_lds_addr = []
        for i in range_constexpr(num_buffers):
            stages_a_lds_addr.append(
                _dg0_lane(make_desc_a(stages_a_mem[i], arith.index(0)), 1)
            )
            stages_b_lds_addr.append(
                _dg0_lane(make_desc_b(stages_b_mem[i], arith.index(0)), 1)
            )
            if const_expr(use_ascale_shuffled_tdm):
                stages_as_lds_addr.append(
                    _dg0_lane(make_desc_as(stages_as_mem[i], arith.index(0)), 1)
                )
            if const_expr(is_mxscale):
                stages_bs_lds_addr.append(
                    _dg0_lane(make_desc_bs(stages_bs_mem[i], arith.index(0)), 1)
                )

        desc_a_init = make_desc_a(stages_a_mem[0], split_k_base)
        desc_b_init = make_desc_b(stages_b_mem[0], split_k_base)
        if const_expr(is_ptpc):
            # No scale TDM for PTPC: alias the scale descriptors/addresses to A/B.
            # Scale waves are predicated off, so these selections are never issued.
            stages_as_lds_addr = stages_a_lds_addr
            stages_bs_lds_addr = stages_b_lds_addr
            desc_as_init = desc_a_init
            desc_bs_init = desc_b_init
        elif const_expr(use_ascale_vgpr):
            # A-scale is not a TDM tensor in the VGPR path. Alias slot 2 so the
            # generic 4-way selector stays well-formed; it is predicated off.
            stages_as_lds_addr = stages_a_lds_addr
            desc_as_init = desc_a_init
            desc_bs_init = make_desc_bs(stages_bs_mem[0], split_k_base)
        else:
            desc_as_init = make_desc_as(stages_as_mem[0], split_k_base)
            desc_bs_init = make_desc_bs(stages_bs_mem[0], split_k_base)

        adv_a_i32 = fx.Int32(tile_k // PACK_FACTOR_A)
        adv_b_i32 = fx.Int32(packed_tile_k_b * 16)
        # 32x4 scale TDM descriptors advance one tile's K-blocks per K-step.
        adv_as_i32 = fx.Int32(
            as32_lds_row_stride
            if use_ascale_shuffled_tdm
            else tile_k // SCALE_BLOCK * wmma_m_rep
        )
        adv_bs_i32 = fx.Int32(
            bs32_lds_row_stride
            if is_mxscale
            else tile_k // SCALE_BLOCK * b_scale_load_rep
        )

        _drop_scale_waves = is_ptpc
        if const_expr(use_ascale_shuffled_tdm):
            _active_wave_limit = min(num_warps, 4)
        elif const_expr(use_ascale_vgpr):
            _active_wave_limit = min(num_warps, 3)
        else:
            _active_wave_limit = 2 if _drop_scale_waves else 4
        active_pred_const = arith.select(
            tdm_wave_id < fx.Int32(_active_wave_limit), fx.Int32(1), fx.Int32(0)
        )

        def _select4(values):
            return _select_wave_tdm_value(values[0], values[1], values[2], values[3])

        def _desc_lanes(descs, lane):
            return [_dg0_lane(desc, lane) for desc in descs]

        def _select_active_tdm(stage_lds_addrs, descs, advs):
            active_stages = [
                _select_wave_tdm_value(
                    stage_lds_addrs[0][i],
                    stage_lds_addrs[1][i],
                    stage_lds_addrs[2][i],
                    stage_lds_addrs[3][i],
                )
                for i in range_constexpr(num_buffers)
            ]
            return (
                active_stages,
                _select4(_desc_lanes(descs, 2)),
                _select4(_desc_lanes(descs, 3)),
                _select4([desc.dgroup1 for desc in descs]),
                _select4(advs),
            )

        if const_expr(use_ascale_shuffled_tdm):
            _tdm_stage_sel = (
                stages_a_lds_addr,
                stages_b_lds_addr,
                stages_as_lds_addr,
                stages_bs_lds_addr,
            )
            _tdm_desc_sel = (desc_a_init, desc_b_init, desc_as_init, desc_bs_init)
            _tdm_adv_sel = (adv_a_i32, adv_b_i32, adv_as_i32, adv_bs_i32)
        elif const_expr(use_ascale_vgpr):
            # wave2 is B-scale; wave3 is a predicated padding slot for the 4-way selector.
            _tdm_stage_sel = (
                stages_a_lds_addr,
                stages_b_lds_addr,
                stages_bs_lds_addr,
                stages_bs_lds_addr,
            )
            _tdm_desc_sel = (desc_a_init, desc_b_init, desc_bs_init, desc_bs_init)
            _tdm_adv_sel = (adv_a_i32, adv_b_i32, adv_bs_i32, adv_bs_i32)
        else:
            _tdm_stage_sel = (
                stages_a_lds_addr,
                stages_b_lds_addr,
                stages_as_lds_addr,
                stages_bs_lds_addr,
            )
            _tdm_desc_sel = (desc_a_init, desc_b_init, desc_as_init, desc_bs_init)
            _tdm_adv_sel = (adv_a_i32, adv_b_i32, adv_as_i32, adv_bs_i32)
        (
            active_stage_lds_addr,
            active_addr_lo,
            active_addr_hi,
            active_dgroup1,
            active_adv_i32,
        ) = _select_active_tdm(_tdm_stage_sel, _tdm_desc_sel, _tdm_adv_sel)
        if const_expr(secondary_scale_tdm):
            if const_expr(two_wave_bscale):
                sec_pred_const = arith.select(tdm_wave_is_a, fx.Int32(1), fx.Int32(0))
                sec_stage_lds_addr = stages_bs_lds_addr
                sec_addr_hi = _dg0_lane(desc_bs_init, 3)
                sec_dgroup1 = desc_bs_init.dgroup1
                sec_adv_i32 = adv_bs_i32
                sec_addr_lo_init = _dg0_lane(desc_bs_init, 2)
            elif const_expr(two_wave_scale):
                sec_pred_const = arith.select(
                    tdm_wave_id < fx.Int32(2), fx.Int32(1), fx.Int32(0)
                )
                sec_stage_lds_addr = [
                    arith.select(
                        tdm_wave_is_a, stages_bs_lds_addr[i], stages_as_lds_addr[i]
                    )
                    for i in range_constexpr(num_buffers)
                ]
                sec_addr_hi = arith.select(
                    tdm_wave_is_a,
                    _dg0_lane(desc_bs_init, 3),
                    _dg0_lane(desc_as_init, 3),
                )
                sec_dgroup1 = arith.select(
                    tdm_wave_is_a, desc_bs_init.dgroup1, desc_as_init.dgroup1
                )
                sec_adv_i32 = arith.select(tdm_wave_is_a, adv_bs_i32, adv_as_i32)
                sec_addr_lo_init = arith.select(
                    tdm_wave_is_a,
                    _dg0_lane(desc_bs_init, 2),
                    _dg0_lane(desc_as_init, 2),
                )
            else:
                # 3-wave compatibility: wave2 carries A-scale, wave0 carries B-scale.
                sec_pred_const = arith.select(tdm_wave_is_a, fx.Int32(1), fx.Int32(0))
                sec_stage_lds_addr = stages_bs_lds_addr
                sec_addr_hi = _dg0_lane(desc_bs_init, 3)
                sec_dgroup1 = desc_bs_init.dgroup1
                sec_adv_i32 = adv_bs_i32
                sec_addr_lo_init = _dg0_lane(desc_bs_init, 2)

        def _pipeline_fence(outstanding=0):
            pipeline_fence(outstanding=outstanding, use_cluster=use_cluster)

        def _pipeline_fence_signal(outstanding=0):
            pipeline_fence_signal(outstanding=outstanding, use_cluster=use_cluster)

        def _issue_active_tdm(load_stage, addr_box, k_prefetch=None, sec_box=None):
            dg0 = _pack_dg0(
                active_pred_const,
                active_stage_lds_addr[load_stage],
                addr_box[0],
                active_addr_hi,
            )
            tdm_ops.tensor_load_2d(tdm_ops.TDMDescriptor2D(dg0, active_dgroup1))
            addr_box[0] = addr_box[0] + active_adv_i32
            if const_expr(secondary_scale_tdm):
                dg0s = _pack_dg0(
                    sec_pred_const,
                    sec_stage_lds_addr[load_stage],
                    sec_box[0],
                    sec_addr_hi,
                )
                tdm_ops.tensor_load_2d(tdm_ops.TDMDescriptor2D(dg0s, sec_dgroup1))
                sec_box[0] = sec_box[0] + sec_adv_i32
            if k_prefetch is not None:
                _l2_prefetch(k_prefetch)

        # Prologue
        if const_expr(secondary_scale_tdm):
            active_sec_lo = sec_addr_lo_init
        for i in range_constexpr(pre_loaded):
            addr_box = [active_addr_lo]
            if const_expr(secondary_scale_tdm):
                sec_box = [active_sec_lo]
                _issue_active_tdm(i, addr_box, sec_box=sec_box)
                active_sec_lo = sec_box[0]
            else:
                _issue_active_tdm(i, addr_box)
            active_addr_lo = addr_box[0]
        _bvs_tail_seed = []
        _bvs_tail_issue_start = loop_iters * num_buffers
        if const_expr(_bvs_active):
            _bvs_initial_depth = _bvs_D if loop_iters > 0 else min(_bvs_D, num_k_tiles)
            _bvs_pf = [
                _bvs_prefetch(split_k_base + arith.index(_d * tile_k))
                for _d in range(_bvs_initial_depth)
            ]
            if const_expr(loop_iters > 0):
                _bvs_ra = [_v for _a in _bvs_pf for _v in _a]
            else:
                _bvs_tail_seed = list(_bvs_pf)
                _bvs_tail_issue_start = _bvs_initial_depth

        _pipeline_fence(outstanding=TDM_LOADS_PER_STEP * (num_buffers - 2))

        # Main loop — acc_mixed style: fence at top, TDM_load mid-compute.
        # This overlaps TDM DMA with the remaining WMMA instructions,
        _fence_outstanding = TDM_LOADS_PER_STEP * (num_buffers - 2)

        if const_expr(loop_iters > 0 and use_tdm_late_signal_overlap):
            _pipeline_fence_signal(outstanding=_fence_outstanding)

        if const_expr(loop_iters > 0):
            init_args = list(accs) + [active_addr_lo]
            if const_expr(secondary_scale_tdm):
                init_args = init_args + [active_sec_lo]
            if const_expr(_bvs_active):
                init_args = init_args + _bvs_ra

            for loop_iter, state in range(0, loop_iters, 1, init=init_args):
                accs_in = list(state[:n_accs])
                cur_addr_lo = state[n_accs]
                _state_off = n_accs + 1
                if const_expr(secondary_scale_tdm):
                    cur_sec_lo = state[_state_off]
                    _state_off = _state_off + 1
                if const_expr(_bvs_active):
                    _ra0 = _state_off
                    _ring_a = list(state[_ra0 : _ra0 + _bvs_D * _vs_tile_a])
                    _state_off = _ra0 + _bvs_D * _vs_tile_a

                for buf_idx in range_constexpr(num_buffers):
                    load_stage = (buf_idx + num_buffers - 1) % num_buffers

                    addr_box = [cur_addr_lo]
                    sec_box = [cur_sec_lo] if secondary_scale_tdm else None

                    def _mid_tdm_ws(
                        _ls=load_stage,
                        _ab=addr_box,
                        _sb=sec_box,
                        _k_off=(
                            split_k_base
                            + loop_iter * arith.index(num_buffers * tile_k)
                            + arith.index(buf_idx * tile_k)
                        ),
                    ):
                        _issue_active_tdm(_ls, _ab, k_prefetch=_k_off, sec_box=_sb)

                    if const_expr(not use_tdm_late_signal_overlap):
                        _pipeline_fence_signal(outstanding=_fence_outstanding)
                    pipeline_fence_wait(use_cluster=use_cluster)

                    _late_tdm_ws_fence_signal = None
                    if const_expr(use_tdm_late_signal_overlap):

                        def _late_tdm_ws_split_signal():
                            _pipeline_fence_signal(outstanding=_fence_outstanding)

                        _late_tdm_ws_fence_signal = _late_tdm_ws_split_signal

                    a0_prefetch = maybe_prefetch_fp8_deep_a0(stages_a_idx[buf_idx])
                    rocdl.sched_barrier(0)
                    if const_expr(_bvs_active):
                        _cur_a = _ring_a[:_vs_tile_a]
                        _next_kb = (
                            split_k_base
                            + loop_iter * arith.index(num_buffers * tile_k)
                            + arith.index((buf_idx + _bvs_D) * tile_k)
                        )
                        _ring_a = _ring_a[_vs_tile_a:] + list(_bvs_prefetch(_next_kb))
                    else:
                        _cur_a = None

                    accs_in = compute_tile_scheduled(
                        accs_in,
                        stages_a_idx[buf_idx],
                        stages_b_idx[buf_idx],
                        stages_as_idx[buf_idx],
                        stages_bs_idx[buf_idx],
                        mid_compute_callback=_mid_tdm_ws,
                        late_compute_callback=_late_tdm_ws_fence_signal,
                        a0_prefetch=a0_prefetch,
                        pf_a_scales=_cur_a,
                    )
                    cur_addr_lo = addr_box[0]
                    if const_expr(secondary_scale_tdm):
                        cur_sec_lo = sec_box[0]
                    hot_loop_scheduler_scheduled()

                _sec_yield = [cur_sec_lo] if secondary_scale_tdm else []
                _bvs_yield = _ring_a if _bvs_active else []
                results = yield list(accs_in) + [cur_addr_lo] + _sec_yield + _bvs_yield

            accs = list(results[:n_accs])
            active_addr_lo = results[n_accs]
            _result_off = n_accs + 1
            if const_expr(secondary_scale_tdm):
                active_sec_lo = results[n_accs + 1]
                _result_off = _result_off + 1
            if const_expr(_bvs_active):
                _bvs_tail_flat = list(
                    results[_result_off : _result_off + _bvs_D * _vs_tile_a]
                )
                _bvs_tail_seed = [
                    _bvs_tail_flat[_d * _vs_tile_a : (_d + 1) * _vs_tile_a]
                    for _d in range(_bvs_D)
                ]
                _bvs_tail_issue_start = loop_iters * num_buffers + _bvs_D
        # Tail — same acc_mixed pattern: fence at top, TDM mid-compute.
        if const_expr(loop_iters > 0 and use_tdm_late_signal_overlap):
            pipeline_fence_wait(use_cluster=use_cluster)
        if const_expr(loop_iters > 0):
            _pipeline_fence(outstanding=0)
        elif const_expr(use_cluster):
            cluster.cluster_barrier()
        epi_addrs_box = [None]
        _ptpc_scale_box = [None]

        def _load_ptpc_scales_once():
            if const_expr(is_ptpc and _ptpc_scale_box[0] is None):
                _ptpc_scale_box[0] = epilogue_load_ptpc_scales()

        _tail_had_load = False
        _bvs_tail_kt = [loop_iters * num_buffers]

        def _bvs_tail_kb():
            if const_expr(not _bvs_active):
                return None
            kb = split_k_base + arith.index(_bvs_tail_kt[0] * tile_k)
            _bvs_tail_kt[0] += 1
            return kb

        _bvs_tail_ring = list(_bvs_tail_seed)
        _bvs_tail_issue_kt = [_bvs_tail_issue_start]

        def _bvs_tail_issue_one():
            if const_expr(_bvs_active and _bvs_tail_issue_kt[0] < num_k_tiles):
                kb = split_k_base + arith.index(_bvs_tail_issue_kt[0] * tile_k)
                _bvs_tail_ring.append(_bvs_prefetch(kb))
                _bvs_tail_issue_kt[0] += 1

        def _bvs_tail_scales():
            if const_expr(_bvs_active):
                return None, _bvs_tail_ring.pop(0)
            return _bvs_tail_kb(), None

        if const_expr(_bvs_active):
            rocdl.sched_barrier(0)

        for _load_stage, _compute_stage, _outstanding in tail_plan:
            _entry_kb, _pf_a_scales = _bvs_tail_scales()
            if const_expr(_outstanding == -1):
                if const_expr(_tail_had_load):
                    _pipeline_fence(outstanding=0)
                if const_expr(tdm_store_enabled):
                    a0_prefetch = maybe_prefetch_fp8_deep_a0(
                        stages_a_idx[_compute_stage]
                    )
                    accs = compute_tile_scheduled(
                        accs,
                        stages_a_idx[_compute_stage],
                        stages_b_idx[_compute_stage],
                        stages_as_idx[_compute_stage],
                        stages_bs_idx[_compute_stage],
                        emit_filler=(_load_ptpc_scales_once if is_ptpc else None),
                        a0_prefetch=a0_prefetch,
                        scale_k_base=_entry_kb,
                        pf_a_scales=_pf_a_scales,
                    )
                else:

                    def _emit_epi_addrs():
                        epi_addrs_box[0] = epilogue_prepare_addrs()
                        _load_ptpc_scales_once()

                    a0_prefetch = maybe_prefetch_fp8_deep_a0(
                        stages_a_idx[_compute_stage]
                    )
                    accs = compute_tile_scheduled(
                        accs,
                        stages_a_idx[_compute_stage],
                        stages_b_idx[_compute_stage],
                        stages_as_idx[_compute_stage],
                        stages_bs_idx[_compute_stage],
                        emit_filler=_emit_epi_addrs,
                        a0_prefetch=a0_prefetch,
                        scale_k_base=_entry_kb,
                        pf_a_scales=_pf_a_scales,
                    )
            else:
                _pipeline_fence_signal(outstanding=_outstanding)
                pipeline_fence_wait(use_cluster=use_cluster)

                _tail_mid_cb = None
                if const_expr(_load_stage is not None):
                    _tail_had_load = True
                    _tail_addr_box = [active_addr_lo]
                    _tail_sec_box = [active_sec_lo] if secondary_scale_tdm else None

                    def _tail_mid_ws(
                        _ls=_load_stage, _ab=_tail_addr_box, _sb=_tail_sec_box
                    ):
                        _issue_active_tdm(_ls, _ab, sec_box=_sb)

                    _tail_mid_cb = _tail_mid_ws

                a0_prefetch = maybe_prefetch_fp8_deep_a0(stages_a_idx[_compute_stage])
                rocdl.sched_barrier(0)
                _bvs_tail_issue_one()
                accs = compute_tile_scheduled(
                    accs,
                    stages_a_idx[_compute_stage],
                    stages_b_idx[_compute_stage],
                    stages_as_idx[_compute_stage],
                    stages_bs_idx[_compute_stage],
                    mid_compute_callback=_tail_mid_cb,
                    a0_prefetch=a0_prefetch,
                    scale_k_base=_entry_kb,
                    pf_a_scales=_pf_a_scales,
                )

                if const_expr(_load_stage is not None):
                    active_addr_lo = _tail_addr_box[0]
                    if const_expr(secondary_scale_tdm):
                        active_sec_lo = _tail_sec_box[0]

                hot_loop_scheduler_scheduled()

        if const_expr(is_ptpc):
            _load_ptpc_scales_once()
            _ptpc_sa, _ptpc_sb = _ptpc_scale_box[0]
            accs = epilogue_apply_ptpc_scale(accs, _ptpc_sa, _ptpc_sb)

        def _emit_tdm_store():
            if const_expr(d_need_epilogue_fence):
                _pipeline_fence(outstanding=0)
            rocdl.sched_barrier(0)
            epilogue_lds_stores(accs, d_lds_buffer, d_lane_base)
            rocdl.s_wait_dscnt(0)
            tdm_ops.tensor_store_2d(d_desc)
            tdm_ops.tensor_wait(0)

        def _emit_buffer_store():
            rocdl.sched_barrier(0)
            if const_expr(epi_addrs_box[0] is None):
                epi_addrs_box[0] = epilogue_prepare_addrs()
            if const_expr(split_k > 1):
                epilogue_atomic_adds(accs, epi_addrs_box[0])
            else:
                epilogue_stores(accs, epi_addrs_box[0])

        if const_expr(tdm_store_enabled):
            full_tile = (blk_m + arith.index(tile_m)) <= m_idx
            if_op = scf.IfOp(full_tile, [], has_else=True)
            with ir.InsertionPoint(if_op.then_block):
                _emit_tdm_store()
                scf.YieldOp([])
            with ir.InsertionPoint(if_op.else_block):
                _emit_buffer_store()
                scf.YieldOp([])
        else:
            _emit_buffer_store()

    cache_tag = (
        data_format,
        scale_mode,
        K,
        tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        num_buffers,
        compute_schedule_kind,
        effective_waves_per_eu,
        l2_prefetch_distance,
        cluster_m,
        cluster_n,
        tdm_store_enabled,
        out_dtype,
        inst_prefetch,
        split_k,
        expert_sched_mode,
        atomic_barrier_enable,
        ascale_load_path,
        _row_major_k_prefetch_depth,
        _bvs_D,
    )

    @flyc.jit
    def launch_mxscale_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_a_scale: fx.Tensor,
        arg_b_scale: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        i32_lda: fx.Int32,
        i32_ldc: fx.Int32,
        stream: fx.Stream,
    ):
        _ = cache_tag
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            arena_alloc.finalized = False
            arena_alloc.finalize()

        gx = (i32_m + (tile_m - 1)) // tile_m
        gy = N // tile_n
        gz = split_k

        if const_expr(use_cluster):
            # Cluster launch needs a cluster-divisible grid
            gx = ((gx + (cluster_m - 1)) // cluster_m) * cluster_m

        cluster_arg = (cluster_m, cluster_n, 1) if use_cluster else None
        kernel_mxscale_gemm(
            arg_c,
            arg_a,
            arg_b,
            arg_a_scale,
            arg_b_scale,
            i32_m,
            i32_n,
            i32_lda,
            i32_ldc,
            value_attrs={
                "rocdl.waves_per_eu": effective_waves_per_eu,
                "rocdl.cluster_dims": (
                    f"{cluster_m},{cluster_n},1" if const_expr(use_cluster) else None
                ),
            },
        ).launch(
            grid=(gx, gy, gz),
            block=(block_threads, 1, 1),
            stream=stream,
            cluster=cluster_arg,
        )

    if effective_expert_sched_mode:
        launch_mxscale_gemm.compile_hints["llvm_options"] = {
            "amdgpu-expert-scheduling-mode": True,
        }

    return launch_mxscale_gemm


def compile_mxscale_gemm(**kw):
    """Backward-compatible wrapper: MX block-scale (E8M0) GEMM."""
    return compile_fp8fp4_gemm(scale_mode="mxscale", **kw)


def compile_mxfp4_gemm(**kw):
    return compile_fp8fp4_gemm(data_format="fp4", scale_mode="mxscale", **kw)


def compile_mxfp8_gemm(**kw):
    return compile_fp8fp4_gemm(data_format="fp8", scale_mode="mxscale", **kw)


def compile_a8w4_gemm(**kw):
    return compile_fp8fp4_gemm(data_format="a8w4", scale_mode="mxscale", **kw)


def compile_ptpc_gemm(
    *,
    N: int = 0,
    K: int,
    data_format: str = "fp8",
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    m_warp: int = 2,
    n_warp: int = 2,
    num_buffers: int = 4,
    waves_per_eu: int = None,
    l2_prefetch_distance: int = 0,
    cluster_m: int = 1,
    cluster_n: int = 1,
    out_dtype: str = "bf16",
    inst_prefetch: bool = False,
    expert_sched_mode: bool = True,
    atomic_barrier_enable: bool = False,
    split_k: int = 1,
):
    """Compile a PTPC (per-token per-channel) GEMM kernel.

    A scale is per-token (sa[M], fp32), B scale is per-channel (sb[N], fp32),
    both constant along K. The K-loop runs the WMMA unscaled (FP8) or with an
    identity E8M0 scale (A8W4, which has no non-scale op); sa*sb is applied in
    the epilogue in fp32. split_k>1 is supported (atomic add path).

    data_format: "fp8" (FP8 act + FP8 weight) or "a8w4" (FP8 act + FP4 weight).
    Requires m_warp*n_warp >= 2 (wave-specialized TDM).
    """
    return compile_fp8fp4_gemm(
        data_format=data_format,
        scale_mode="ptpc",
        N=N,
        K=K,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        m_warp=m_warp,
        n_warp=n_warp,
        num_buffers=num_buffers,
        waves_per_eu=waves_per_eu,
        l2_prefetch_distance=l2_prefetch_distance,
        cluster_m=cluster_m,
        cluster_n=cluster_n,
        out_dtype=out_dtype,
        inst_prefetch=inst_prefetch,
        expert_sched_mode=expert_sched_mode,
        atomic_barrier_enable=atomic_barrier_enable,
        split_k=split_k,
    )


__all__ = [
    "compile_fp8fp4_gemm",
    "compile_mxscale_gemm",
    "compile_mxfp4_gemm",
    "compile_mxfp8_gemm",
    "compile_a8w4_gemm",
    "compile_ptpc_gemm",
]
