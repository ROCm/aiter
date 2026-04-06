"""A8W8 FP8 GEMM kernel for gfx1250 with per-token scales and optional bias.

Computes: C = (A @ B^T) * (a_scale * b_scale) + bias

Where:
    A: [M, K] FP8 (row-major, contiguous in K)
    B: [N, K] FP8 (row-major, contiguous in K)
    a_scale: [M] float32 per-token scale
    b_scale: [N] float32 per-channel scale
    bias: [N] float (optional)
    C: [M, N] output (bf16/f16/f32)

Uses WMMA 16x16x128 FP8 instructions with identity E8M0 scales (1.0),
then applies per-token/per-channel scales in the epilogue.
Supports 2/3/4-stage TDM async pipelining for loads.
Output is written via buffer_store (no TDM store).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx

from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl, tdm_ops, vector
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from flydsl.expr import idx2crd
from .gemm_gfx1250_common import (
    enable_wmma_pipeline, get_lds_memref, lds_load_b128,
    pipeline_fence, store_acc_vec8_to_buffer,
)
from .pipeline_utils import make_tail_plan

# WMMA tile dimensions for FP8
WMMA_M, WMMA_N, WMMA_K = 16, 16, 128
WAVE_SIZE = 32 #numthreads

# FP8 WMMA operand: 16 VGPRs (vec<16xi32>)
# Each lane holds 64 bytes = 64 FP8 elements (one K-half of 128)
DS_LOADS_PER_FRAG = 4  # 4 x ds_load_b128 = 64 bytes per lane

# LDS padding in bytes (4 DWORDs = 16 bytes)
LDS_PAD_A_BYTES = 16
LDS_PAD_B_BYTES = 16
# E8M0 identity scale: 127 = 2^(127-127) = 2^0 = 1.0
E8M0_IDENTITY = 0x7F7F7F7F  # 4 packed E8M0(127) values in i32

_STAGE_NAMES = ("ping", "pong", "pang", "pung")


def compile_gemm_a8w8(
    *,
    K: int,
    tile_m: int = 128,
    tile_n: int = 256,
    tile_k: int = 128,
    m_warp: int = 2,
    n_warp: int = 4,
    num_buffers: int = 2,
    cluster_m: int = 1,
    cluster_n: int = 1,
    waves_per_eu: int = None,
    l2_prefetch_distance: int = 0,
    use_tdm_load: bool = True,
    out_dtype: str = "bf16",
    has_bias: bool = False,
):
    """Compile an A8W8 FP8 GEMM kernel with per-token scales.

    Computes: C = (A @ B^T) * (a_scale * b_scale) [+ bias]

    Args:
        K: Inner dimension (must be known at compile time).
        tile_m: Block tile size along M.
        tile_n: Block tile size along N.
        tile_k: Block tile size along K.
        m_warp: Number of warps along M dimension.
        n_warp: Number of warps along N dimension.
        num_buffers: Number of LDS pipeline stages (2, 3, or 4).
        waves_per_eu: Occupancy hint (None=default).
        l2_prefetch_distance: K-tiles ahead to prefetch into L2 (0=disabled).
        out_dtype: Output element type ("bf16", "f16", "f32").
        has_bias: Whether to add bias vector after scaling.

    Returns:
        JitFunction: launch_fn(C, A, B, a_scale, b_scale, [bias,] M, N, stream)
    """
    if out_dtype not in ("f32", "bf16", "f16"):
        raise ValueError(f"out_dtype must be 'f32', 'bf16', or 'f16', got {out_dtype!r}")
    elem_bytes_d = 2 if out_dtype in ("bf16", "f16") else 4 # needed later for byte offsets in buffer_store

    if num_buffers not in (2, 3, 4):
        raise ValueError(f"num_buffers must be 2, 3, or 4, got {num_buffers}")

    use_cluster = cluster_m > 1 or cluster_n > 1
    if use_cluster:
        if cluster_m * cluster_n > 16:
            raise ValueError(
                f"cluster_m * cluster_n must be <= 16, got {cluster_m}*{cluster_n}")

    effective_waves_per_eu = waves_per_eu
    if use_cluster and effective_waves_per_eu is None:
        effective_waves_per_eu = 1

    num_warps = m_warp * n_warp
    block_threads = num_warps * WAVE_SIZE

    if tile_k % WMMA_K != 0:
        raise ValueError(f"tile_k must be a multiple of {WMMA_K}, got {tile_k}")
    if tile_m % WMMA_M != 0:
        raise ValueError(f"tile_m must be a multiple of {WMMA_M}, got {tile_m}")
    if tile_n % WMMA_N != 0:
        raise ValueError(f"tile_n must be a multiple of {WMMA_N}, got {tile_n}")

    warp_tile_m = tile_m // m_warp # M chunk handled by each warp
    warp_tile_n = tile_n // n_warp # N chunk handled by each warp
    if warp_tile_m % WMMA_M != 0:
        raise ValueError(f"warp_tile_m={warp_tile_m} must be a multiple of {WMMA_M}")
    if warp_tile_n % WMMA_N != 0:
        raise ValueError(f"warp_tile_n={warp_tile_n} must be a multiple of {WMMA_N}")

    if K % tile_k != 0:
        raise ValueError(f"K must be divisible by tile_k={tile_k}, got K={K}")
    num_k_tiles = K // tile_k
    if num_k_tiles < num_buffers:
        raise ValueError(
            f"{num_buffers}-stage buffering requires num_k_tiles >= {num_buffers}, "
            f"got {num_k_tiles}")

    gpu_arch = str(get_hip_arch(timeout_s=300))
    assert gpu_arch.startswith("gfx1250"), f"Expected gfx1250, got {gpu_arch}"

    k_wmma_steps = tile_k // WMMA_K
    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N
    n_accs = wmma_m_rep * wmma_n_rep # each warp needs n_accs accumulator vectors (vec<8xf32> each accumulator vec is 8 VGPRs) 

    # LDS layout: FP8 data stored as bytes, accessed via f16 memrefs because you can't declare memref<sizexf8>
    lds_a_stride_bytes = tile_k + LDS_PAD_A_BYTES
    lds_b_stride_bytes = tile_k + LDS_PAD_B_BYTES
    lds_a_stride = lds_a_stride_bytes // 2  # in f16 elements
    lds_b_stride = lds_b_stride_bytes // 2

    lds_a_data_bytes = tile_m * lds_a_stride_bytes # lds needed for tile A
    lds_b_data_bytes = tile_n * lds_b_stride_bytes # lds needed for tile B

    # Allocate LDS for each pipeline stage
    stage_allocators = []
    stage_a_data_off = []
    stage_b_data_off = []
    
    # each stage has a separate memref.global symbol. mlir backend places them at non overlapping lds addresses
    # stage_a_data_off and stage_b_data_off are byte offsets within each stage's allocation, not absolute LDS addresses
    for i in range(num_buffers):
        name = _STAGE_NAMES[i]
        alloc = SmemAllocator(None, arch=gpu_arch, global_sym_name=f"a8w8_{name}")

        off = alloc._align(alloc.ptr, 16) # align to 16 bytes
        stage_a_data_off.append(off) # would be 0
        alloc.ptr = off + lds_a_data_bytes # advance pointer past A

        off = alloc._align(alloc.ptr, 16)
        stage_b_data_off.append(off)
        alloc.ptr = off + lds_b_data_bytes # advance pointer past B

        stage_allocators.append(alloc)

    # Pipeline plan
    pre_loaded = num_buffers - 1
    loop_iters = (num_k_tiles - pre_loaded) // num_buffers # I think this is kinda loop unrolling?
    _tail_start = loop_iters * num_buffers
    extra = num_k_tiles - _tail_start - pre_loaded
    tail_plan = make_tail_plan(num_buffers, pre_loaded, extra)

    @flyc.kernel
    def kernel_gemm_a8w8(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_a_scale: fx.Tensor,
        arg_b_scale: fx.Tensor,
        arg_bias: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
    ):
        enable_wmma_pipeline() # Enable back-to-back WMMA issue

        tx = gpu.thread_id("x") # thread index within workgroup (total 32 thrds/warp *8 warps = 256 thrds)
        bx = gpu.block_id("x")
        by = gpu.block_id("y")

        blk_m = bx * arith.index(tile_m) # starting M row for the workgroup
        blk_n = by * arith.index(tile_n) # starting N col for the workgroup

        # MCAST masks for cluster TDM loads
        if use_cluster:
            local_x, local_y = gpu.compute_cluster_position()
            a_mcast_mask, b_mcast_mask = gpu.compute_mcast_masks(
                local_x, local_y, cluster_m, cluster_n)
        else:
            a_mcast_mask = 0
            b_mcast_mask = 0

        # Thread/wave decomposition (same as wmma_gemm)
        layout_thr = fx.make_layout(
            (m_warp, n_warp, 2, 16),
            (n_warp * WAVE_SIZE, WAVE_SIZE, 16, 1))
        thr_coord = idx2crd(tx, layout_thr) # convert thread ID into a 4D coordinate
        wave_m_idx, wave_n_idx, lane_kgrp, lane16 = (
            fx.get(thr_coord, 0), fx.get(thr_coord, 1),
            fx.get(thr_coord, 2), fx.get(thr_coord, 3))

        warp_m_base = wave_m_idx * arith.index(warp_tile_m) # starting row for this warp
        warp_n_base = wave_n_idx * arith.index(warp_tile_n) # starting column for this warp

        # Buffer resources for output and scales
        m_idx = arith.index_cast(T.index, i32_m.ir_value())
        n_idx = arith.index_cast(T.index, i32_n.ir_value())
        n_stride = arith.index_cast(T.index, i32_n.ir_value())
        c_nrec = m_idx * n_stride * arith.index(elem_bytes_d) # total bytes in C
        c_rsrc = buffer_ops.create_buffer_resource(arg_c, num_records_bytes=c_nrec)

        # Scale buffer resources (f32, 4 bytes per element)
        a_scale_nrec = m_idx * arith.index(4)
        a_scale_rsrc = buffer_ops.create_buffer_resource(
            arg_a_scale, num_records_bytes=a_scale_nrec)
        b_scale_nrec = n_idx * arith.index(4)
        b_scale_rsrc = buffer_ops.create_buffer_resource(
            arg_b_scale, num_records_bytes=b_scale_nrec)

        if has_bias:
            bias_nrec = n_idx * arith.index(4)
            bias_rsrc = buffer_ops.create_buffer_resource(
                arg_bias, num_records_bytes=bias_nrec)

        # Identity E8M0 scale for WMMA (2^0 = 1.0)
        identity_scale = arith.constant(E8M0_IDENTITY, type=T.i32)

        elem_ty_lds = T.f16 # LDS memref element type

        # --- TDM async copy helpers ---
        # cache_policy=1 enables streaming/non-temporal for large matrices
        _b_cache_policy = 1

        def copy_a_to_lds(k_base, lds_mem_ref):
            desc = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_a, # base ptr of A in global mem 
                lds_memref=lds_mem_ref, # destination LDS memref (ping, pong, ... etc.)
                global_offset=(blk_m, k_base), # offsets
                tensor_shape=(tile_m, tile_k),
                strides=(K, 1),
                tile_shape=(tile_m, tile_k),
                elem_bytes=1, # FP8
                pad_interval=tile_k, pad_amount=LDS_PAD_A_BYTES, # pad_interval should be 256 (need to check this)
                num_warps=num_warps,
                workgroup_mask=a_mcast_mask)
            tdm_ops.tensor_load_2d(desc)

        def copy_b_to_lds(k_base, lds_mem_ref):
            desc = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_b, lds_memref=lds_mem_ref,
                global_offset=(blk_n, k_base),
                tensor_shape=(tile_n, tile_k),
                strides=(K, 1),
                tile_shape=(tile_n, tile_k),
                elem_bytes=1,
                pad_interval=tile_k, pad_amount=LDS_PAD_B_BYTES,
                num_warps=num_warps,
                workgroup_mask=b_mcast_mask,
                cache_policy=_b_cache_policy)
            tdm_ops.tensor_load_2d(desc)

        def issue_all_tdm_loads(k_base, a_mem, b_mem):
            rocdl.s_setprio(2)
            copy_a_to_lds(k_base, a_mem)
            copy_b_to_lds(k_base, b_mem)
            rocdl.s_setprio(0)

        # --- Non-TDM copy: buffer_load + vector.store to LDS ---
        # A[M,K] is row-major FP8 (1 byte). LDS layout: tile_m rows, lds_a_stride_bytes cols.
        # Each thread copies vec4 i16 = 8 bytes = 8 FP8 elements at a time.
        _a_copy_elems = tile_m * tile_k  # total FP8 elements in tile
        _a_vec_size = 8  # 8 bytes per copy = vec4 i16
        _a_total_vecs = _a_copy_elems // _a_vec_size
        _a_vecs_per_thread = (_a_total_vecs + block_threads - 1) // block_threads

        _b_copy_elems = tile_n * tile_k
        _b_total_vecs = _b_copy_elems // _a_vec_size
        _b_vecs_per_thread = (_b_total_vecs + block_threads - 1) // block_threads

        def _copy_to_lds_sync(k_base, a_smem_ptr, b_smem_ptr):
            """Copy A and B tiles from global to LDS using buffer_load + vector.store."""
            a_rsrc = buffer_ops.create_buffer_resource(arg_a, max_size=True)
            b_rsrc = buffer_ops.create_buffer_resource(arg_b, max_size=True)
            a_lds = get_lds_memref(a_smem_ptr)
            b_lds = get_lds_memref(b_smem_ptr)

            # FP8 data = 1 byte/elem. buffer_load with dtype=i16 auto-scales offset by 2,
            # so we pass offset in i16 elements (= byte_offset / 2).
            for t in range_constexpr(_a_vecs_per_thread):
                vec_idx = tx + arith.index(t * block_threads)
                vecs_per_row = tile_k // _a_vec_size
                a_row = vec_idx / arith.index(vecs_per_row)
                a_col_v = vec_idx % arith.index(vecs_per_row)

                # Global offset in i16 elements: byte_off / 2
                g_byte_off = (blk_m + a_row) * arith.index(K) + (k_base + a_col_v * arith.index(_a_vec_size))
                g_off_i16 = g_byte_off / arith.index(2)
                v = buffer_ops.buffer_load(a_rsrc, g_off_i16, vec_width=4, dtype=T.i16)
                v_f16 = vector.bitcast(T.vec(4, T.f16), v)
                lds_off = a_row * arith.index(lds_a_stride) + a_col_v * arith.index(4)
                vector.store(v_f16, a_lds, [lds_off])

            for t in range_constexpr(_b_vecs_per_thread):
                vec_idx = tx + arith.index(t * block_threads)
                vecs_per_row = tile_k // _a_vec_size
                b_row = vec_idx / arith.index(vecs_per_row)
                b_col_v = vec_idx % arith.index(vecs_per_row)

                g_byte_off = (blk_n + b_row) * arith.index(K) + (k_base + b_col_v * arith.index(_a_vec_size))
                g_off_i16 = g_byte_off / arith.index(2)
                v = buffer_ops.buffer_load(b_rsrc, g_off_i16, vec_width=4, dtype=T.i16)
                v_f16 = vector.bitcast(T.vec(4, T.f16), v)
                lds_off = b_row * arith.index(lds_b_stride) + b_col_v * arith.index(4)
                vector.store(v_f16, b_lds, [lds_off])

            gpu.barrier()

        # --- Fragment loading (FP8: 4 x ds_load_b128 -> vec<16xi32>) ---
        def _precompute_a_lane_bases(lds_ptr): # for each warp get what elems each thread would load
            lds_buffer = get_lds_memref(lds_ptr)
            row_base = (warp_m_base + lane16) * arith.index(lds_a_stride)
            k_half_off = lane_kgrp * arith.index(32)  # 32 f16 elems = 64 bytes
            bases = []
            for wm in range_constexpr(wmma_m_rep):
                base = row_base + arith.index(wm * WMMA_M * lds_a_stride) + k_half_off
                bases.append(base)
            return lds_buffer, bases

        def load_a_frag(lds_buffer, a_lane_base, ks): # ks selects which K-subtile. e.g say tile_K=256 then there would be 2 K-subtiles cuz wmma_k=128
            """Load one 16x128 FP8 A-fragment from LDS -> vec<16xi32>."""
            k_elem_off = arith.index(ks * WMMA_K // 2) # //2 because fp16 units
            elem_off = a_lane_base + k_elem_off
            v0 = lds_load_b128(lds_buffer, elem_off)
            v1 = lds_load_b128(lds_buffer, elem_off + arith.index(8))
            v2 = lds_load_b128(lds_buffer, elem_off + arith.index(16))
            v3 = lds_load_b128(lds_buffer, elem_off + arith.index(24))
            v01 = vector.shuffle(v0, v1, list(range(8))) # concatination
            v23 = vector.shuffle(v2, v3, list(range(8)))
            return vector.shuffle(v01, v23, list(range(16)))

        def _precompute_b_lane_bases(lds_ptr): # same shit but for b
            lds_buffer = get_lds_memref(lds_ptr)
            row_base = (warp_n_base + lane16) * arith.index(lds_b_stride)
            k_half_off = lane_kgrp * arith.index(32)
            bases = []
            for wn in range_constexpr(wmma_n_rep):
                base = row_base + arith.index(wn * WMMA_N * lds_b_stride) + k_half_off
                bases.append(base)
            return lds_buffer, bases

        def load_b_frag(lds_buffer, b_lane_base, ks):
            """Load one 128x16 FP8 B-fragment from LDS. Same layout as A."""
            return load_a_frag(lds_buffer, b_lane_base, ks)

        # --- K-subtile compute (A-streaming with identity scales) ---
        def _load_b_frags(b_buf, b_bases, ks): # load all B frags upfront
            return [load_b_frag(b_buf, b_bases[wn], ks)
                    for wn in range_constexpr(wmma_n_rep)]

        # we loaded all B frags upfront then A frags are loaded one at a time
        def _a_streaming_compute(accs, a_buf, a_bases, b_frags, ks,
                                 emit_filler=None, next_b_info=None):
            """Stream A fragments per-wm, interleaved with WMMA"""
            next_b_frags = None
            a_frag = load_a_frag(a_buf, a_bases[0], ks)
            for wm in range_constexpr(wmma_m_rep):
                is_last = (wm == wmma_m_rep - 1)
                if not is_last:
                    a_next = load_a_frag(a_buf, a_bases[wm + 1], ks)
                if is_last:
                    rocdl.s_wait_dscnt(0)
                    
                    # this basically lets us compute the output addresses using the ALU while the last WMMA is going on
                    if emit_filler is not None:
                        rocdl.sched_barrier(0) # tells the compiler that all instructions before this point must be issued before any instrucntions after thsi point
                        emit_filler() # calls epilogue_prepare_addrs() and computes the output addresses for the final store
                    if next_b_info is not None: # prefetching for next K-subtile
                        nb_buf, nb_bases, nb_ks = next_b_info
                        next_b_frags = _load_b_frags(nb_buf, nb_bases, nb_ks)
                else:
                    rocdl.s_wait_dscnt(DS_LOADS_PER_FRAG)
                for wn in range_constexpr(wmma_n_rep):
                    idx = wm * wmma_n_rep + wn
                    # Use wmma_scale with identity E8M0 scales (1.0)
                    accs[idx] = rocdl.wmma_scale_f32_16x16x128_f8f6f4(
                        T.vec(8, T.f32),
                        b_frags[wn], a_frag, accs[idx],
                        identity_scale, identity_scale,
                        fmtA=0, fmtB=0,        # FP8 format
                        scaleAType=0, scaleBType=0,  # E8M0 scales
                    )
                if not is_last:
                    a_frag = a_next
            if next_b_info is not None:
                return accs, next_b_frags
            return accs

        # --- Compute one K-tile ---
        def compute_tile(accs_in, lds_a_ptr, lds_b_ptr, emit_filler=None):
            current_accs = list(accs_in)
            a_buf, a_bases = _precompute_a_lane_bases(lds_a_ptr)
            b_buf, b_bases = _precompute_b_lane_bases(lds_b_ptr)

            if k_wmma_steps == 1:
                b_frags = _load_b_frags(b_buf, b_bases, 0)
                current_accs = _a_streaming_compute(
                    current_accs, a_buf, a_bases, b_frags, 0,
                    emit_filler=emit_filler)
            else:
                prev_b = _load_b_frags(b_buf, b_bases, 0)
                for ks in range_constexpr(k_wmma_steps - 1):
                    current_accs, prev_b = _a_streaming_compute(
                        current_accs, a_buf, a_bases, prev_b, ks,
                        next_b_info=(b_buf, b_bases, ks + 1))
                current_accs = _a_streaming_compute(
                    current_accs, a_buf, a_bases, prev_b,
                    k_wmma_steps - 1, emit_filler=emit_filler)

            return current_accs

        def hot_loop_scheduler():
            rocdl.sched_barrier(0)

        # --- Epilogue: apply per-token scales and optional bias ---
        _half_out = out_dtype in ("bf16", "f16")
        _out_elem = T.bf16 if out_dtype == "bf16" else (T.f16 if out_dtype == "f16" else None)

        def _buffer_load_f32x8(rsrc, base_off):
            """Load 8 consecutive f32 values via two 128-bit buffer loads."""
            lo = buffer_ops.buffer_load(rsrc, base_off, vec_width=4, dtype=T.f32)
            hi = buffer_ops.buffer_load(
                rsrc, base_off + arith.index(4), vec_width=4, dtype=T.f32)
            return vector.shuffle(lo, hi, list(range(8)))

        def apply_scales_and_bias(accs):
            """Apply per-token a_scale and per-channel b_scale, then optional bias.

            Each accumulator vec<8xf32> corresponds to an 8-element sub-tile.
            WMMA layout: lane16 selects the M-row, lane_kgrp*8 + [0..7] selects N-cols.
            """
            for wm in range_constexpr(wmma_m_rep):
                # Load a_scale for this row (scalar f32)
                row_idx = blk_m + warp_m_base + arith.index(wm * WMMA_M) + lane16
                a_sc = buffer_ops.buffer_load(
                    a_scale_rsrc, row_idx, vec_width=1, dtype=T.f32) # load one f32 scale for this lane's row

                for wn in range_constexpr(wmma_n_rep):
                    idx = wm * wmma_n_rep + wn
                    col_base = blk_n + warp_n_base + arith.index(wn * WMMA_N) \
                        + lane_kgrp * arith.index(8)

                    # Load b_scale for these 8 columns (2 x vec4 f32)
                    b_sc_vec = _buffer_load_f32x8(b_scale_rsrc, col_base)

                    # scale = a_scale * b_scale (broadcast a_scale across 8 cols)
                    a_sc_vec = vector.broadcast(T.vec(8, T.f32), a_sc)
                    scale_vec = arith.mulf(a_sc_vec, b_sc_vec)

                    # acc *= scale
                    accs[idx] = arith.mulf(accs[idx], scale_vec)

                    # Optional bias (2 x vec4 f32)
                    if has_bias:
                        bias_vec = _buffer_load_f32x8(bias_rsrc, col_base)
                        accs[idx] = arith.addf(accs[idx], bias_vec)

            return accs

        def epilogue_prepare_addrs():
            addrs = []
            for wm in range_constexpr(wmma_m_rep):
                for wn in range_constexpr(wmma_n_rep):
                    row = blk_m + warp_m_base + arith.index(wm * WMMA_M) + lane16
                    col_base = (blk_n + warp_n_base + arith.index(wn * WMMA_N)
                                + lane_kgrp * arith.index(8))
                    if _half_out:
                        c_off_bytes = (row * n_stride + col_base) * arith.index(elem_bytes_d)
                        addrs.append(c_off_bytes)
                    else:
                        for half in range_constexpr(2):
                            col = col_base + arith.index(half * 4)
                            c_off = row * n_stride + col
                            addrs.append(c_off)
            return addrs

        def epilogue_stores(final_accs, addrs):
            addr_idx = 0
            for wm in range_constexpr(wmma_m_rep):
                for wn in range_constexpr(wmma_n_rep):
                    idx = wm * wmma_n_rep + wn
                    if _half_out:
                        addr_idx += store_acc_vec8_to_buffer(
                            final_accs[idx], c_rsrc, addrs[addr_idx],
                            out_elem=_out_elem, offset_is_bytes=True)
                    else:
                        addr_idx += store_acc_vec8_to_buffer(
                            final_accs[idx], c_rsrc, addrs[addr_idx:addr_idx + 2])

        _effective_l2_pf = l2_prefetch_distance

        def _l2_prefetch(k_base):
            if _effective_l2_pf <= 0:
                return
            pf_k = k_base + arith.index(_effective_l2_pf * tile_k)
            tdm_ops.l2_prefetch_tile(
                arg_a, (blk_m, pf_k), (tile_m, tile_k), (K, 1),
                elem_bytes=1, thread_id=tx, block_threads=block_threads)
            tdm_ops.l2_prefetch_tile(
                arg_b, (blk_n, pf_k), (tile_n, tile_k), (K, 1),
                elem_bytes=1, thread_id=tx, block_threads=block_threads)

        # ====== Multi-stage pipeline ======
        acc_zero = arith.constant_vector(0.0, T.vec(8, T.f32))
        accs = [acc_zero] * n_accs

        lds_a_data_f16 = lds_a_data_bytes // 2
        lds_b_data_f16 = lds_b_data_bytes // 2

        base_ptrs = [sa.get_base() for sa in stage_allocators]

        stages_a = [
            SmemPtr(base_ptrs[i], stage_a_data_off[i], elem_ty_lds,
                    shape=(lds_a_data_f16,))
            for i in range_constexpr(num_buffers)
        ]
        stages_b = [
            SmemPtr(base_ptrs[i], stage_b_data_off[i], elem_ty_lds,
                    shape=(lds_b_data_f16,))
            for i in range_constexpr(num_buffers)
        ]

        # get the actual MLIR memref value to pass to TDM later
        stages_a_mem = [stages_a[i].get() for i in range_constexpr(num_buffers)]
        stages_b_mem = [stages_b[i].get() for i in range_constexpr(num_buffers)]

        if use_tdm_load:
            # ===== TDM async pipeline =====
            # Pattern: load → compute → fence (tensor_wait + barrier).
            # The fence ensures the next buffer is ready and all waves sync
            # before buffer reuse.
            _main_outstanding = 2 * (num_buffers - 2)

            # Prologue: load first (num_buffers - 1) tiles
            for i in range_constexpr(pre_loaded):
                issue_all_tdm_loads(
                    arith.index(i * tile_k),
                    stages_a_mem[i], stages_b_mem[i])
            pipeline_fence(outstanding=_main_outstanding,
                           use_cluster=use_cluster) # basically adding the tensorwait at the start of the loop

            # Main loop
            main_end = loop_iters * num_buffers * tile_k

            if loop_iters > 0:
                for iv, state in range(0, main_end, num_buffers * tile_k, init=list(accs)):
                    rocdl.iglp_opt(1) # instruction level parallelism hint
                    accs_in = list(state)
                    for s in range_constexpr(num_buffers):
                        _load_stage = (s + num_buffers - 1) % num_buffers
                        _load_k_off = (s + num_buffers - 1) * tile_k
                        issue_all_tdm_loads(
                            iv + arith.index(_load_k_off),
                            stages_a_mem[_load_stage], stages_b_mem[_load_stage])
                        accs_in = compute_tile(accs_in, stages_a[s], stages_b[s])
                        hot_loop_scheduler() # rocdl.sched_barrier(0) compiler directive telling it not to reorder instructions before or after this

                        pipeline_fence(outstanding=_main_outstanding,
                                       use_cluster=use_cluster)
                    results = yield list(accs_in)
                accs = list(results)

            # Tail
            # if loop_iters == 0 and use_cluster:
            #     gpu.cluster_barrier() # we don't need this?
            _extra_j = 0 # used to compute extra K-offset for each extra load
            epi_addrs_box = [None] # to capture epilogue address calculation we do during the last compute stage
            for _load_stage, _compute_stage, _outstanding in tail_plan:
                if _load_stage is not None:
                    _k_off = (_tail_start + pre_loaded + _extra_j) * tile_k
                    issue_all_tdm_loads(
                        arith.index(_k_off),
                        stages_a_mem[_load_stage], stages_b_mem[_load_stage])
                    _extra_j += 1
                if _outstanding == -1:
                    def _emit_epi_addrs():
                        epi_addrs_box[0] = epilogue_prepare_addrs()

                    accs = compute_tile(
                        accs, stages_a[_compute_stage], stages_b[_compute_stage],
                        emit_filler=_emit_epi_addrs)
                else:
                    accs = compute_tile(
                        accs, stages_a[_compute_stage], stages_b[_compute_stage])
                    hot_loop_scheduler()
                    pipeline_fence(outstanding=_outstanding,
                                   use_cluster=use_cluster)

        else:
            # ===== Non-TDM synchronous pipeline =====
            # Simple loop: load → barrier → compute for each K tile.
            # Always use stage 0 LDS buffers (no double buffering needed).
            for kblk in range_constexpr(num_k_tiles):
                k_base = arith.index(kblk * tile_k)
                _copy_to_lds_sync(k_base, stages_a[0], stages_b[0])
                accs = compute_tile(accs, stages_a[0], stages_b[0])
                gpu.barrier()

            epi_addrs_box = [None]

        # Apply per-token scales and optional bias
        accs = apply_scales_and_bias(accs)

        # Store results
        if epi_addrs_box[0] is None:
            epi_addrs_box[0] = epilogue_prepare_addrs()
        epilogue_stores(accs, epi_addrs_box[0])

    cache_tag = (K, tile_m, tile_n, tile_k, m_warp, n_warp,
                 num_buffers, cluster_m, cluster_n,
                 effective_waves_per_eu, l2_prefetch_distance,
                 use_tdm_load, out_dtype, has_bias)

    @flyc.jit
    def launch_gemm_a8w8(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_a_scale: fx.Tensor,
        arg_b_scale: fx.Tensor,
        arg_bias: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        stream: fx.Stream,
    ):
        # needed to build the cache key so that compile_gemm_a8w8 knows if these values changed and it can't use the cached binary
        _ = cache_tag
        
        # finalize lds allocation (so far we only know how much LDS we need thanks to SmemAllocator objects we created earlier but now we emit it into the MLIR)
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body): #ensures these land inside the gpu.module
            for alloc in stage_allocators:
                alloc.finalized = False # reset needed incase this gets traced more than once (idk why this can happen but it can)
            for alloc in stage_allocators:
                alloc.finalize()

        # grid dimensions (just tracing MLIR no execution happening so grid size is not known at compile time)
        idx_m = arith.index_cast(T.index, i32_m.ir_value()) # get the raw SSA and cast it to fx.index type. i32_m is in an fx.Int32 wrapper
        idx_n = arith.index_cast(T.index, i32_n.ir_value())
        gx = _raw((idx_m + arith.index(tile_m - 1)) / arith.index(tile_m))
        gy = _raw((idx_n + arith.index(tile_n - 1)) / arith.index(tile_n)) # unwrap to get the raw SSA

        # trigger tracing of the flyc.kernel function. builds the MLIR gpu.func
        launcher = kernel_gemm_a8w8(
            arg_c, arg_a, arg_b, arg_a_scale, arg_b_scale, arg_bias,
            i32_m, i32_n)
        
        # walk the MLIR module and find the gpu.func then attach the HW hints
        # the attributes are added after tracing because they're metadata about the kernel, not ops inside it
        for op in ctx.gpu_module_body.operations:
            if hasattr(op, 'attributes') and op.OPERATION_NAME == "gpu.func":
                if effective_waves_per_eu is not None:
                    _wpe = int(effective_waves_per_eu)
                    if _wpe >= 1:
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                            ir.IntegerType.get_signless(32), _wpe) #gpu.func @kernel_gemm_a8w8(...) attributes {rocdl.waves_per_eu = 1 : i32}    
                if use_cluster:
                    op.attributes["rocdl.cluster_dims"] = ir.StringAttr.get(
                        f"{cluster_m},{cluster_n},1")
        cluster_arg = (cluster_m, cluster_n, 1) if use_cluster else None
        
        # emits gpu.launch_func into the MLIR module
        launcher.launch(
            grid=(gx, gy, 1), # runtime SSA values computed above
            block=(block_threads, 1, 1), # compile time constant
            stream=stream,
            cluster=cluster_arg,
        )

    return launch_gemm_a8w8


__all__ = ["compile_gemm_a8w8"]
