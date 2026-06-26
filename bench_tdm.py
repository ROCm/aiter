#!/usr/bin/env python3
"""TDM-load latency ablation for the a8w4/mxscale gfx1250 GEMM (clean-room).

Goal: isolate how much of the small-M a8w4 GEMM latency is pure TDM global->LDS
data movement, per input tensor. Rather than ablate inside the production kernel
(historical baggage / hidden interactions), this builds a *from-scratch* mini
kernel that reproduces the production load structure exactly:

  - wave-specialized TDM: B is split across wave0 (B_half0) and wave1
    (B_half1), A uses wave0 (shared with B_half0), B_scale uses wave2.
    Each active wave issues its descriptor;
  - num_buffers-deep LDS ring with the production fence pattern
    (signal/wait + tensor_wait), descriptors rebuilt per K-tile;
  - NO LDS->VGPR fragment loads, NO WMMA. Just the loads + fences.
  - production small-M A-scale route variants use raw scale buffer_load -> VGPR
    with invalid M lanes predicated off and a small prefetch ring. The optional
    *_sync variants keep the old immediate consume path as a worst-case diagnostic.

A compile-time mask enables each tensor independently, so we can measure:
  none / a / b0 / b1 / b(b0+b1) / bs / ab / all.

Anti-DCE: `tensor_load_2d` is a side-effecting intrinsic (never DCE'd), but as
belt-and-suspenders each active loader wave reads one ds_load_b128 from every
LDS stage it wrote and the wave leader stores an xor-reduction into C. The
stored value is meaningless; it only keeps the LDS observable.

Inputs (shapes/precision/preshuffle) are built identically to the production
a8w4 mxscale test, so the TDM descriptors move byte-for-byte the same traffic
(for tile-aligned shapes; this kernel does NOT host-pad, see the shape asserts).

Caveat: descriptors are rebuilt per K-tile via the high-level builder, whereas the
production hot path packs dgroup0 once and increments the address per tile. The
*traffic/descriptor semantics* match, but the per-tensor marginal may include a
small amount of scalar descriptor-rebuild work the production path avoids. For the
M=1 shape this is ~6 builds/tensor of a few index ops each, overlapped with the
async TDM DMA + fences, so it is negligible vs the µs-scale marginals.

Timing: EAGER mode + L2 flush (COLD cache) via the latest production timer
`_bench_kernel_us(run_once, flush_cache=...)`, which clears a `--flush-mb` MiB
scratch buffer before each timed launch so every load reads from HBM cold
(defeats the gfx1250 MALL/Infinity-Cache). One variant per subprocess to avoid
clock-throttling contamination across heavy back-to-back kernels.
"""

import argparse
import functools
import inspect
import math
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../FlyDSL"))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

import flydsl  # noqa: E402,F401
import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from flydsl._mlir import ir  # noqa: E402
from flydsl._mlir.dialects import scf  # noqa: E402
from flydsl.compiler.kernel_function import CompilationContext  # noqa: E402
from flydsl.expr import (  # noqa: E402
    arith,
    buffer_ops,
    const_expr,
    gpu,
    range_constexpr,
    rocdl,
    tdm_ops,
    vector,
)
from flydsl.expr.typing import T  # noqa: E402
from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr  # noqa: E402
from kernels.gemm_common_gfx1250 import (  # noqa: E402
    extract_lds_base_idx,
    lds_load_b128_raw,
    lds_store_b128,
    pipeline_fence,
    pipeline_fence_signal,
    pipeline_fence_wait,
)

# Production timer + input builders (identical shapes/preshuffle to the real test).
# We use the EAGER + L2-flush cold-cache timer from the latest harness:
#   _bench_kernel_us(run_once, flush_cache=..., warmup, iters)
# which clears `flush_cache` (scratch MiB) before each timed launch so the kernel
# reads from HBM cold (defeats the gfx1250 MALL/Infinity-Cache).
from tests.kernels.test_gemm_fp8fp4_gfx1250 import (  # noqa: E402
    SCALE_BLOCK,
    _bench_kernel_us,
    _bench_kernel_us_cudagraph,
    preshuffle_e8m0_scale,
    random_fp8_data,
)
from tests.kernels.utils import fp4_utils  # noqa: E402

WMMA_M, WMMA_N, WMMA_K = 16, 16, 128
WAVE_SIZE = 32
LDS_PAD_A_BYTES = 16
_SCALE_GUARD_BYTES = 16


def _vec_chunks(n: int):
    chunks = []
    done = 0
    while done < n:
        w = 4 if (n - done) >= 4 else (2 if (n - done) >= 2 else 1)
        chunks.append((done, w))
        done += w
    return chunks


# Tensor -> default loader-wave id.
# B is split across wave0 (b0 = first half of tile_n) and wave1 (b1 = second half).
# A-scale is NOT loaded via TDM (removed); B-scale uses wave2.
TENSORS = ("a", "b0", "b1", "bs")
WAVE_OF = {"a": 0, "b0": 0, "b1": 1, "bs": 2}

_TDM_SIG_PARAMS = set(inspect.signature(tdm_ops.make_tensor_descriptor_2d).parameters)


def _make_tdm_desc(**kwargs):
    filtered = {k: v for k, v in kwargs.items() if k in _TDM_SIG_PARAMS}
    return tdm_ops.make_tensor_descriptor_2d(**filtered)


def _align_up(value, align):
    return (value + align - 1) // align * align


def _select_ascale_load_path(M: int) -> str:
    return "vgpr" if M < 32 else "shuffled_tdm"


def _prepare_a_scale_for_path(
    a_scale: torch.Tensor, ascale_load_path: str, warp_tile_m: int, scale_k_per_tile: int, tile_m: int,
) -> torch.Tensor:
    if ascale_load_path == "vgpr":
        return a_scale
    if ascale_load_path == "shuffled_tdm":
        return preshuffle_e8m0_scale(a_scale, warp_tile_m, scale_k_per_tile=scale_k_per_tile, row_align=tile_m)
    raise ValueError(f"unsupported ascale_load_path={ascale_load_path!r}")


@functools.lru_cache(maxsize=64)
def compile_tdm_load_ablation(
    *,
    N: int,
    K: int,
    load_mask: tuple,  # subset of TENSORS, sorted
    desc_only: bool = False,
    desc_only_lds: bool = False,
    vgpr_ascale: bool = False,
    vgpr_ascale_mode: str = "prefetch",
    tile_m: int = 16,
    tile_n: int = 64,
    tile_k: int = 512,
    m_warp: int = 1,
    n_warp: int = 4,
    num_buffers: int = 4,
    waves_per_eu: int = None,
    expert_sched_mode: bool = True,
    b_layout: str = "preshuffle",
):
    """Compile the a8w4/mxscale TDM-load ablation mini kernel.

    load_mask selects which of {a, b0, b1, bs} are loaded; everything else
    (LDS->VGPR, WMMA, real store) is omitted. B is split across wave0/wave1.
    desc_only keeps the same descriptor-building control flow but suppresses
    the TDM issue.

    b_layout: 'preshuffle' uses preshuffle_b_16x16 with dim0=packed_tile_k_b*16
              (wide contiguous rows, few outer rows).
              'tiled' uses preshuffle_b_16x16_tiled (ksmajor=False) with dim0=256
              (flat block array, each block 256B; n-group-major so N-half
              split is contiguous).
    """
    gpu_arch = str(get_rocm_arch())
    if not gpu_arch.startswith("gfx1250"):
        raise RuntimeError(f"Expected gfx1250, got {gpu_arch}")

    load = set(load_mask)
    assert load.issubset(set(TENSORS)), load_mask
    if vgpr_ascale_mode not in ("prefetch", "sync"):
        raise ValueError(f"vgpr_ascale_mode must be 'prefetch' or 'sync', got {vgpr_ascale_mode!r}")

    def loader_wave_of(t):
        return WAVE_OF[t]

    # ---- a8w4 / mxscale compile-time constants (clean subset of the real kernel) ----
    PACK_A, PACK_B = 1, 2  # FP8 activation, FP4 weight
    num_warps = m_warp * n_warp
    block_threads = num_warps * WAVE_SIZE
    # wave-specialized: each loaded tensor needs its loader wave to exist.
    need_waves = max([loader_wave_of(t) for t in load], default=-1) + 1
    if num_warps < need_waves:
        raise ValueError(f"load_mask {load_mask} needs >= {need_waves} waves, got {num_warps}")

    # Shape validation: the descriptors assume tile-aligned, packable shapes (the
    # production path host-pads in _run_benchmark; this clean-room kernel does NOT
    # pad, so reject unaligned shapes loudly instead of silently truncating/OOB).
    if num_buffers < 2:
        raise ValueError(f"num_buffers must be >= 2, got {num_buffers}")
    if K % tile_k != 0:
        raise ValueError(f"K={K} must be divisible by tile_k={tile_k}")
    if tile_k % SCALE_BLOCK != 0:
        raise ValueError(f"tile_k={tile_k} must be divisible by SCALE_BLOCK={SCALE_BLOCK}")
    if tile_k % WMMA_K != 0:
        raise ValueError(f"tile_k={tile_k} must be divisible by WMMA_K={WMMA_K}")
    if tile_k % PACK_B != 0:
        raise ValueError(f"tile_k={tile_k} must be divisible by PACK_B={PACK_B}")
    if tile_m % WMMA_M != 0 or tile_n % WMMA_N != 0:
        raise ValueError(f"tile_m={tile_m}/tile_n={tile_n} must be multiples of {WMMA_M}/{WMMA_N}")
    if N % 32 != 0 or tile_n % 32 != 0:
        raise ValueError(f"32x4 B-scale layout requires N%32==0 and tile_n%32==0; got N={N}, tile_n={tile_n}")
    if N % tile_n != 0:
        raise ValueError(f"N={N} must be divisible by tile_n={tile_n} (no host pad here)")
    if N % 16 != 0 or K % 16 != 0:
        raise ValueError(f"N={N}, K={K} must be multiples of 16 (B 16x16 preshuffle)")
    num_k_tiles = K // tile_k
    if num_k_tiles < num_buffers:
        raise ValueError(f"{num_buffers}-buf requires num_k_tiles >= {num_buffers}, got {num_k_tiles}")

    packed_tile_k_a = tile_k // PACK_A
    packed_tile_k_b = tile_k // PACK_B
    K_packed_b = K // PACK_B
    K_scale = K // SCALE_BLOCK
    scale_k_per_tile = tile_k // SCALE_BLOCK
    k_wmma_steps = tile_k // WMMA_K

    warp_tile_m = tile_m // m_warp
    wmma_m_rep = warp_tile_m // WMMA_M
    ascale_opsel = vgpr_ascale and wmma_m_rep >= 2 and (wmma_m_rep & (wmma_m_rep - 1)) == 0
    ascale_half = wmma_m_rep // 2
    ascale_load = ascale_half if ascale_opsel else wmma_m_rep

    # B-split: each half loads tile_n/2 rows of B.
    ab_split_b_groups = tile_n // 32  # groups per half (in units of 16-row preshuffle blocks)
    B_TILE_BYTES = 256  # one 16x16 preshuffle block
    _b_num_blocks = (tile_n // 16) * (packed_tile_k_b // 16)  # blocks per full GEMM tile
    _b_num_blocks_half = _b_num_blocks // 2  # blocks per B half
    if b_layout == "tiled":
        b_half_bytes = _b_num_blocks_half * B_TILE_BYTES
    else:
        b_half_bytes = (tile_n // 2) * packed_tile_k_b

    # B-scale TDM layout: 32 rows x 4 K-scales, packed as [ceil(rows/32), (K//128)*128].
    bs32_block_bytes = 128
    bs32_global_row_stride = (K // WMMA_K) * bs32_block_bytes
    bs32_lds_row_stride = k_wmma_steps * bs32_block_bytes
    bs32_tile_blocks = tile_n // 32
    bs32_tile_blocks_pad = 1 << (bs32_tile_blocks - 1).bit_length()

    # ---- per-stage LDS region sizes (bytes) for enabled tensors only ----
    lds_a_stride_bytes = packed_tile_k_a + LDS_PAD_A_BYTES
    region_bytes = {
        "a": tile_m * lds_a_stride_bytes,
        "b0": b_half_bytes,
        "b1": b_half_bytes,
        "bs": bs32_tile_blocks_pad * bs32_lds_row_stride + _SCALE_GUARD_BYTES,
    }

    # Stage layout: contiguous enabled regions, 16B aligned, per-stage pitch 128B.
    rel_off = {}
    ptr = 0
    for t in TENSORS:
        if t in load:
            ptr = _align_up(ptr, 16)
            rel_off[t] = ptr
            ptr += region_bytes[t]
    stage_bytes = _align_up(max(ptr, 16), 128)
    stage_off = [s * stage_bytes for s in range(num_buffers)]
    arena_bytes = stage_bytes * num_buffers

    arena = SmemAllocator(
        None, arch=gpu_arch,
        global_sym_name=(
            f"tdm_ablation_{''.join(sorted(load))}"
            f"{'_desc' if desc_only else ''}{'_desc_lds' if desc_only_lds else ''}"
            f"_{tile_m}x{tile_n}x{tile_k}_{num_buffers}b_{b_layout}"
        ),
    )
    arena.ptr = arena_bytes

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def kernel_tdm_ablation(
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
        rocdl.disable_xdl_arb_stall()
        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        by = gpu.block_id("y")
        wave_id = rocdl.wave_id()
        lane = arith.index_cast(T.index, tx) % arith.index(WAVE_SIZE)
        lane16 = lane % arith.index(16)
        lane_kgrp = lane / arith.index(16)

        blk_m = bx * arith.index(tile_m)
        blk_n = by * arith.index(tile_n)
        m_idx = fx.Index(i32_m)
        lda_packed = fx.Index(i32_lda)  # PACK_A == 1

        arena_base = arena.get_base()
        elem_ty_lds = T.f16  # SmemPtr shapes count f16 elements (bytes // 2)

        def _load_contig_i32(rsrc, base_idx, n, soff):
            out = [None] * n
            chunks = _vec_chunks(n)
            for ci in range_constexpr(len(chunks)):
                start, width = chunks[ci]
                off = arith.index_cast(T.i32, base_idx + arith.index(start))
                raw = buffer_ops.buffer_load(rsrc, off, vec_width=width, dtype=T.i32, soffset_bytes=soff)
                if const_expr(width == 1):
                    out[start] = raw
                else:
                    vec = fx.Vector(raw)
                    for c in range_constexpr(width):
                        out[start + c] = vec[c]
            return out

        scale_identity_i32 = arith.constant(0x7F7F7F7F, type=T.i32)
        if const_expr(vgpr_ascale):
            ascale_nbytes = m_idx * arith.index(K_scale)
            ascale_rsrc = buffer_ops.create_buffer_resource(
                arg_a_scale,
                max_size=False,
                num_records_bytes=ascale_nbytes,
            )
            ascale_row_i32 = K_scale // 4
            ascale_row0 = blk_m + lane16
            if const_expr(ascale_opsel):
                ascale_row0 = ascale_row0 + lane_kgrp * arith.index(ascale_half * WMMA_M)

            def _load_contig_i32_guarded_row(row, n, soff):
                row_valid = row < m_idx
                if_op = scf.IfOp(row_valid, [T.i32] * n, has_else=True)
                with ir.InsertionPoint(if_op.then_block):
                    vals = _load_contig_i32(
                        ascale_rsrc,
                        row * arith.index(ascale_row_i32),
                        n,
                        soff,
                    )
                    scf.YieldOp([arith.unwrap(v) for v in vals])
                with ir.InsertionPoint(if_op.else_block):
                    scf.YieldOp([arith.unwrap(scale_identity_i32) for _ in range(n)])
                return list(if_op.results)

            def load_ascale_vgpr(k_base):
                kt = k_base / arith.index(tile_k)
                soff = arith.index_cast(T.i32, kt * arith.index(scale_k_per_tile))
                full_tile = (blk_m + arith.index(tile_m)) <= m_idx
                vals = [None] * (k_wmma_steps * ascale_load)
                for i in range_constexpr(ascale_load):
                    row = ascale_row0 + arith.index(i * WMMA_M)
                    if_op = scf.IfOp(full_tile, [T.i32] * k_wmma_steps, has_else=True)
                    with ir.InsertionPoint(if_op.then_block):
                        row_vals = _load_contig_i32(
                            ascale_rsrc,
                            row * arith.index(ascale_row_i32),
                            k_wmma_steps,
                            soff,
                        )
                        scf.YieldOp([arith.unwrap(v) for v in row_vals])
                    with ir.InsertionPoint(if_op.else_block):
                        row_vals = _load_contig_i32_guarded_row(row, k_wmma_steps, soff)
                        scf.YieldOp([arith.unwrap(v) for v in row_vals])
                    for ks in range_constexpr(k_wmma_steps):
                        vals[ks * ascale_load + i] = if_op.results[ks]
                return vals

            vgpr_red_box = [arith.constant(0, type=T.i32)]

            def sink_ascale_vgpr(vals):
                for i in range_constexpr(len(vals)):
                    vgpr_red_box[0] = arith.xori(vgpr_red_box[0], vals[i])

        # Per-tensor, per-stage LDS handles (only for enabled tensors).
        stage_mem = {t: [] for t in load}
        stage_idx = {t: [] for t in load}
        for t in load:
            n_f16 = region_bytes[t] // 2
            for s in range_constexpr(num_buffers):
                p = SmemPtr(arena_base, stage_off[s] + rel_off[t], elem_ty_lds, shape=(n_f16,))
                stage_mem[t].append(p.get())
                stage_idx[t].append(extract_lds_base_idx(p))

        # ---- descriptor builders (a8w4 specialization of the production kernel) ----
        def desc_a(memref, k_base):
            return _make_tdm_desc(
                global_ptr=arg_a, lds_memref=memref,
                global_offset=(blk_m, k_base),
                tensor_shape=(tile_m, packed_tile_k_a),
                strides=(K // PACK_A, 1),
                tile_shape=(tile_m, packed_tile_k_a),
                elem_bytes=1, pad_interval=packed_tile_k_a, pad_amount=LDS_PAD_A_BYTES,
                num_warps=1, early_timeout=True, oob_outer_bound=i32_m,
            )

        def desc_b_half(memref, k_base, n_half):
            if const_expr(b_layout == "tiled"):
                _kt = K_packed_b // packed_tile_k_b
                bn = blk_n / arith.index(tile_n)
                bk = k_base / arith.index(tile_k)
                tile_blk0 = (bn * arith.index(_kt) + bk) * arith.index(_b_num_blocks)
                half_blk0 = tile_blk0 + arith.index(n_half * _b_num_blocks_half)
                return _make_tdm_desc(
                    global_ptr=arg_b, lds_memref=memref,
                    global_offset=(half_blk0, arith.index(0)),
                    tensor_shape=(N * K_packed_b // B_TILE_BYTES, B_TILE_BYTES),
                    strides=(B_TILE_BYTES, 1),
                    tile_shape=(_b_num_blocks_half, B_TILE_BYTES),
                    elem_bytes=1, pad_interval=0, pad_amount=0,
                    num_warps=1, early_timeout=True,
                )
            else:
                k_packed_off = k_base / arith.index(PACK_B)
                group_start = n_half * ab_split_b_groups
                return _make_tdm_desc(
                    global_ptr=arg_b, lds_memref=memref,
                    global_offset=(blk_n / arith.index(16) + arith.index(group_start), k_packed_off * arith.index(16)),
                    tensor_shape=(N // 16, K_packed_b * 16),
                    strides=(K_packed_b * 16, 1),
                    tile_shape=(ab_split_b_groups, packed_tile_k_b * 16),
                    elem_bytes=1, pad_interval=0, pad_amount=0,
                    num_warps=1, early_timeout=True,
                )

        def desc_b0(memref, k_base):
            return desc_b_half(memref, k_base, 0)

        def desc_b1(memref, k_base):
            return desc_b_half(memref, k_base, 1)

        def desc_bs(memref, k_base):
            block_off = blk_n / arith.index(32)
            col_off = (k_base / arith.index(WMMA_K)) * arith.index(bs32_block_bytes)
            return _make_tdm_desc(
                global_ptr=arg_b_scale, lds_memref=memref,
                global_offset=(block_off, col_off),
                tensor_shape=(N // 32, bs32_global_row_stride),
                strides=(bs32_global_row_stride, 1),
                tile_shape=(bs32_tile_blocks_pad, bs32_lds_row_stride),
                elem_bytes=1, pad_interval=0, pad_amount=0,
                num_warps=1, early_timeout=True, oob_outer_bound=N // 32,
            )

        DESC = {"a": desc_a, "b0": desc_b0, "b1": desc_b1, "bs": desc_bs}

        def sink_desc(desc, loader_slot):
            dg0 = fx.Vector(desc.dgroup0)
            dg1 = fx.Vector(desc.dgroup1)
            red = arith.constant(0, type=T.i32)
            for j in range_constexpr(4):
                red = arith.xori(red, dg0[j])
            for j in range_constexpr(8):
                red = arith.xori(red, dg1[j])
            gx = (m_idx + arith.index(tile_m - 1)) / arith.index(tile_m)
            linear_wg = by * gx + bx
            is_leader = arith.cmpi(
                arith.CmpIPredicate.eq, arith.index_cast(T.i32, lane), arith.constant(0, type=T.i32)
            )
            out_rsrc = buffer_ops.create_buffer_resource(arg_c, max_size=False)
            store_if = scf.IfOp(is_leader)
            with ir.InsertionPoint(store_if.then_block):
                voff = arith.index_cast(
                    T.i32, linear_wg * arith.index(num_warps) + arith.index(loader_slot)
                )
                buffer_ops.buffer_store(red, out_rsrc, voff)
                scf.YieldOp([])

        def sink_desc_lds(desc, t):
            """LDS-only descriptor sink: XOR descriptor fields into LDS stage 0.

            Same SALU as sink_desc (builds descriptor, computes XOR) but writes to
            LDS instead of global C. Eliminates the buffer_store and s_wait_xcnt
            overhead so that b_desc_lds - none measures pure descriptor-build SALU.
            """
            dg0 = fx.Vector(desc.dgroup0)
            dg1 = fx.Vector(desc.dgroup1)
            red = arith.constant(0, type=T.i32)
            for j in range_constexpr(4):
                red = arith.xori(red, dg0[j])
            for j in range_constexpr(8):
                red = arith.xori(red, dg1[j])
            is_leader = arith.cmpi(
                arith.CmpIPredicate.eq, arith.index_cast(T.i32, lane), arith.constant(0, type=T.i32)
            )
            store_if = scf.IfOp(is_leader)
            with ir.InsertionPoint(store_if.then_block):
                # Write 16 bytes (replicate red) to stage 0 of this tensor's LDS region.
                sink_vec = vector.from_elements(T.vec(4, T.i32), [red, red, red, red])
                lds_store_b128(stage_mem[t][0], arith.index(0), sink_vec)
                scf.YieldOp([])

        def issue_tdm_stage(stage, k_base):
            # Each enabled tensor issues its full-tile descriptor on its loader wave.
            for t in TENSORS:
                if const_expr(t in load):
                    is_loader = arith.cmpi(
                        arith.CmpIPredicate.eq, wave_id, arith.constant(loader_wave_of(t), type=T.i32)
                    )
                    if_op = scf.IfOp(is_loader)
                    with ir.InsertionPoint(if_op.then_block):
                        desc = DESC[t](stage_mem[t][stage], k_base)
                        if const_expr(desc_only):
                            sink_desc(desc, loader_wave_of(t))
                        elif const_expr(desc_only_lds):
                            sink_desc_lds(desc, t)
                        else:
                            tdm_ops.tensor_load_2d(desc)
                        scf.YieldOp([])

        def issue_stage(stage, k_base):
            issue_tdm_stage(stage, k_base)
            if const_expr(vgpr_ascale and vgpr_ascale_mode == "sync"):
                sink_ascale_vgpr(load_ascale_vgpr(k_base))

        # ---- num_buffers-deep software pipeline ----
        # Prologue: pre-issue all num_buffers stages to fill the ring.
        # Main loop: fence(outstanding=num_buffers-1), consume oldest, issue next.
        # Tail loop: drain with decreasing outstanding (num_buffers-1 .. 0).

        if const_expr(vgpr_ascale and vgpr_ascale_mode == "prefetch"):
            vgpr_ring = []
            vgpr_issue_kt = [0]

            def issue_vgpr_one():
                if const_expr(vgpr_issue_kt[0] < num_k_tiles):
                    vgpr_ring.append(load_ascale_vgpr(arith.index(vgpr_issue_kt[0] * tile_k)))
                    vgpr_issue_kt[0] += 1

            def consume_vgpr_one():
                if const_expr(len(vgpr_ring) > 0):
                    sink_ascale_vgpr(vgpr_ring.pop(0))

        # Prologue: fill all num_buffers slots.
        for i in range_constexpr(min(num_buffers, num_k_tiles)):
            issue_stage(i % num_buffers, arith.index(i * tile_k))
            if const_expr(vgpr_ascale and vgpr_ascale_mode == "prefetch"):
                issue_vgpr_one()

        pipeline_fence(outstanding=num_buffers - 1)
        # Main loop: steady state with num_buffers in flight.
        # fence(num_buffers-1) waits until oldest completes, then consume + issue next.
        for t_idx in range_constexpr(num_buffers, num_k_tiles):
            stage = t_idx % num_buffers
            if const_expr(vgpr_ascale and vgpr_ascale_mode == "prefetch"):
                consume_vgpr_one()
            # oldest buffer is now free; reuse it for the next tile
            issue_stage(stage, arith.index(t_idx * tile_k))
            if const_expr(vgpr_ascale and vgpr_ascale_mode == "prefetch"):
                issue_vgpr_one()
            pipeline_fence(outstanding=num_buffers - 1)

        # Tail loop: drain remaining in-flight buffers one at a time.
        for i in range_constexpr(min(num_buffers, num_k_tiles)):
            remaining = min(num_buffers, num_k_tiles) - 1 - i
            pipeline_fence(outstanding=remaining)
            if const_expr(vgpr_ascale and vgpr_ascale_mode == "prefetch"):
                consume_vgpr_one()

        # ---- anti-DCE sink: each loader wave reads its LDS stages, leader stores ----
        # tensor_load_2d is opaque side-effecting (never DCE'd); this reads offset 0
        # of every stage (always in-bounds, >= 16 B) so the LDS is provably observed.
        # Each workgroup writes a DISTINCT slot via the linear block id
        # (by*grid_x + bx) so the checksum proves every WG ran and there is no
        # same-address write race across the gy=ceil(N/tile_n) blocks.
        if const_expr(len(load) > 0 and not desc_only and not desc_only_lds):
            gx = (m_idx + arith.index(tile_m - 1)) / arith.index(tile_m)
            linear_wg = by * gx + bx
            is_leader = arith.cmpi(
                arith.CmpIPredicate.eq, arith.index_cast(T.i32, lane), arith.constant(0, type=T.i32)
            )
            out_rsrc = buffer_ops.create_buffer_resource(arg_c, max_size=False)
            for t in TENSORS:
                if const_expr(t in load):
                    is_loader = arith.cmpi(
                        arith.CmpIPredicate.eq, wave_id, arith.constant(loader_wave_of(t), type=T.i32)
                    )
                    if_op = scf.IfOp(is_loader)
                    with ir.InsertionPoint(if_op.then_block):
                        red = arith.constant(0, type=T.i32)
                        for s in range_constexpr(num_buffers):
                            v = fx.Vector(lds_load_b128_raw(stage_idx[t][s], arith.index(0)))
                            for j in range_constexpr(4):
                                red = arith.xori(red, vector.extract(v, static_position=[j], dynamic_position=[]))
                        store_if = scf.IfOp(is_leader)
                        with ir.InsertionPoint(store_if.then_block):
                            voff = arith.index_cast(
                                T.i32, linear_wg * arith.index(num_warps) + arith.index(loader_wave_of(t))
                            )
                            buffer_ops.buffer_store(red, out_rsrc, voff)
                            scf.YieldOp([])
                        scf.YieldOp([])

        if const_expr(vgpr_ascale):
            gx = (m_idx + arith.index(tile_m - 1)) / arith.index(tile_m)
            linear_wg = by * gx + bx
            is_leader = arith.cmpi(
                arith.CmpIPredicate.eq, arith.index_cast(T.i32, lane), arith.constant(0, type=T.i32)
            )
            out_rsrc = buffer_ops.create_buffer_resource(arg_c, max_size=False)
            store_if = scf.IfOp(is_leader)
            with ir.InsertionPoint(store_if.then_block):
                wave_idx = arith.index_cast(T.index, wave_id)
                voff = arith.index_cast(
                    T.i32, linear_wg * arith.index(num_warps * 2) + arith.index(num_warps) + wave_idx
                )
                buffer_ops.buffer_store(vgpr_red_box[0], out_rsrc, voff)
                scf.YieldOp([])

    cache_tag = (
        N, K, tuple(sorted(load)), desc_only, desc_only_lds, vgpr_ascale, vgpr_ascale_mode,
        tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers, b_layout,
    )

    @flyc.jit
    def launch(arg_c, arg_a, arg_b, arg_a_scale, arg_b_scale,
               i32_m, i32_n, i32_lda, i32_ldc, stream):
        _ = cache_tag
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            arena.finalized = False
            arena.finalize()
        gx = (i32_m + (tile_m - 1)) // tile_m
        gy = (i32_n + (tile_n - 1)) // tile_n
        kernel_tdm_ablation(
            arg_c, arg_a, arg_b, arg_a_scale, arg_b_scale,
            i32_m, i32_n, i32_lda, i32_ldc,
            value_attrs={"rocdl.waves_per_eu": waves_per_eu},
        ).launch(grid=(gx, gy, 1), block=(block_threads, 1, 1), stream=stream)

    if expert_sched_mode:
        launch.compile_hints["llvm_options"] = {"amdgpu-expert-scheduling-mode": True}
    return launch


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
# Tuple: (load_mask, vgpr_ascale, vgpr_ascale_mode, desc_only, desc_only_lds)
# desc_only     = build descriptor, sink to global C (s_wait_xcnt exposed)
# desc_only_lds = build descriptor, sink to LDS only (isolates pure SALU cost)
VARIANTS = {
    "none":             ((), False, "prefetch", False, False),
    "a":                (("a",), False, "prefetch", False, False),
    "b0":               (("b0",), False, "prefetch", False, False),
    "b1":               (("b1",), False, "prefetch", False, False),
    "b":                (("b0", "b1"), False, "prefetch", False, False),
    "b_desc":           (("b0", "b1"), False, "prefetch", True,  False),
    "b_desc_lds":       (("b0", "b1"), False, "prefetch", False, True),
    "bs":               (("bs",), False, "prefetch", False, False),
    "ab":               (("a", "b0", "b1"), False, "prefetch", False, False),
    "all":              (("a", "b0", "b1", "bs"), False, "prefetch", False, False),
    "asv":              ((), True, "prefetch", False, False),
    "b_asv":            (("b0", "b1"), True, "prefetch", False, False),
    "b_asv_desc":       (("b0", "b1"), True, "prefetch", True,  False),
    "scale_prod":       (("bs",), True, "prefetch", False, False),
    "all_prod":         (("a", "b0", "b1", "bs"), True, "prefetch", False, False),
    "all_prod_desc":    (("a", "b0", "b1", "bs"), True, "prefetch", True,  False),
    "asv_sync":         ((), True, "sync", False, False),
    "b_asv_sync":       (("b0", "b1"), True, "sync", False, False),
    "scale_prod_sync":  (("bs",), True, "sync", False, False),
    "all_prod_sync":    (("a", "b0", "b1", "bs"), True, "sync", False, False),
}


def _build_inputs(M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, *,
                   ascale_load_path=None, b_layout="preshuffle"):
    """a8w4/mxscale inputs, matching the current production test layouts."""
    torch.manual_seed(0)
    PACK_B = 2
    a = random_fp8_data(M, K)                      # FP8 activation [M, K]
    b = fp4_utils.random_fp4_packed(N, K)          # FP4 weight [N, K/2]
    a_scale = fp4_utils.random_e8m0(M, K // SCALE_BLOCK)
    b_scale = fp4_utils.random_e8m0(N, K // SCALE_BLOCK)

    ascale_load_path = ascale_load_path or _select_ascale_load_path(M)
    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp
    skt = tile_k // SCALE_BLOCK
    as_ps = _prepare_a_scale_for_path(a_scale, ascale_load_path, warp_tile_m, skt, tile_m)
    bs_ps = preshuffle_e8m0_scale(b_scale, warp_tile_n, scale_k_per_tile=skt)
    if b_layout == "tiled":
        b_ps = fp4_utils.preshuffle_b_16x16_tiled(
            b, N, K // PACK_B, tile_n=tile_n, tile_kb=tile_k // PACK_B, ksmajor=False,
        )
    else:
        b_ps = fp4_utils.preshuffle_b_16x16(b, N, K // PACK_B)

    c = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    return (
        c, a.cuda(), b_ps.cuda(), as_ps.cuda(), bs_ps.cuda(),
    )


def _bench_one(fn, bufs, M, N, K, warmup, iters, flush_mb, method="eager", num_slots=0):
    """Cold-cache timing. method='eager' -> per-iter L2 flush (models the real single
    launch). method='graph' -> hipGraph; num_slots>1 rotates distinct input buffers
    (cold via rotation, GPU kept continuously busy = boosted clock)."""
    c, a, b, asc, bsc = bufs
    stream = torch.cuda.current_stream()
    # Compile to a replayable launcher (the bare JIT fn re-dispatches each call).
    compiled_exe = flyc.compile(fn, c, a, b, asc, bsc, M, N, K, N, stream)

    def run():
        compiled_exe(c, a, b, asc, bsc, M, N, K, N, torch.cuda.current_stream())

    c.zero_()
    run()
    torch.cuda.synchronize()
    # Raw-byte integer reduction: the sink stores random i32 bit patterns into a
    # bf16 buffer, so a float reduction would hit inf/nan. Sum of raw bytes is
    # finite and non-zero iff some WG wrote -> valid "kernel ran" proof.
    checksum = int(c.view(torch.uint8).to(torch.int64).sum().item())

    if method == "graph":
        us = _bench_kernel_us_cudagraph(run, warmup=warmup, iters=iters)
    else:
        us = _bench_kernel_us(run, flush_l2=True, warmup=warmup, iters=iters)
    return us, checksum


def _measure_single(name, args):
    """Compile + time exactly one variant (or 'real') in this fresh process.

    One measurement per process avoids the clock-throttling contamination seen
    when many heavy graph captures run back-to-back (heavier kernels inflate
    more: empty ~1.8x, full GEMM ~4.6x after 8 variants).
    """
    import time

    os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "1"
    arch = str(get_rocm_arch())
    if arch != "gfx1250":
        raise SystemExit(f"requires gfx1250, got {arch}")

    if name == "real":
        from kernels.gemm_fp8fp4_gfx1250 import compile_mxscale_gemm
        M, N, K = args.M, args.N, args.K
        ascale_load_path = _select_ascale_load_path(M)
        bufs = _build_inputs(
            M, N, K, args.tile_m, args.tile_n, args.tile_k, args.m_warp, args.n_warp,
            ascale_load_path=ascale_load_path,
        )
        fn = compile_mxscale_gemm(
            data_format="a8w4", N=N, K=K,
            tile_m=args.tile_m, tile_n=args.tile_n, tile_k=args.tile_k,
            m_warp=args.m_warp, n_warp=args.n_warp, num_buffers=args.num_buffers,
            out_dtype="bf16", l2_prefetch_distance=0, ascale_load_path=ascale_load_path,
        )
        loaded = "a+b+bs+ascale_vgpr+wmma+store" if ascale_load_path == "vgpr" else "a+b+as+bs+wmma+store"
    else:
        M, N, K = args.M, args.N, args.K
        mask, vgpr_ascale, vgpr_ascale_mode, desc_only, desc_only_lds = VARIANTS[name]
        mask = tuple(sorted(mask))
        ascale_load_path = "vgpr" if vgpr_ascale else _select_ascale_load_path(M)
        bufs = _build_inputs(
            M, N, K, args.tile_m, args.tile_n, args.tile_k, args.m_warp, args.n_warp,
            ascale_load_path=ascale_load_path, b_layout=args.b_layout,
        )
        fn = compile_tdm_load_ablation(
            N=N, K=K, load_mask=mask, desc_only=desc_only, desc_only_lds=desc_only_lds,
            vgpr_ascale=vgpr_ascale, vgpr_ascale_mode=vgpr_ascale_mode,
            tile_m=args.tile_m, tile_n=args.tile_n, tile_k=args.tile_k,
            m_warp=args.m_warp, n_warp=args.n_warp, num_buffers=args.num_buffers,
            b_layout=args.b_layout,
        )
        loaded_parts = list(mask)
        if desc_only:
            loaded_parts.append("desc_only")
        if desc_only_lds:
            loaded_parts.append("desc_lds_only")
        if vgpr_ascale:
            loaded_parts.append(f"as_vgpr_{vgpr_ascale_mode}")
        loaded = "+".join(loaded_parts) if loaded_parts else "(none)"

    time.sleep(0.5)  # let clocks settle before timing
    us, checksum = _bench_one(fn, bufs, M, N, K, args.warmup, args.iters, args.flush_mb,
                              method=args.method, num_slots=args.num_slots)
    print(f"RESULT,{name},{loaded},{us:.4f},{checksum:.6e}", flush=True)


def _sweep(args):
    import subprocess

    names = [n.strip() for n in args.variants.split(",")]
    for n in names:
        if n not in VARIANTS:
            raise SystemExit(f"unknown variant {n!r}; choices: {','.join(VARIANTS)}")
    if args.with_real_gemm:
        names.append("real")

    shape_args = [
        "-M", str(args.M), "-N", str(args.N), "-K", str(args.K),
        "--tile-m", str(args.tile_m), "--tile-n", str(args.tile_n), "--tile-k", str(args.tile_k),
        "--m-warp", str(args.m_warp), "--n-warp", str(args.n_warp),
        "--num-buffers", str(args.num_buffers),
        "--warmup", str(args.warmup), "--iters", str(args.iters),
        "--flush-mb", str(args.flush_mb), "--method", args.method,
        "--num-slots", str(args.num_slots),
        "--b-layout", args.b_layout,
    ]

    if args.method == "graph":
        timer = f"GRAPH rotating buffers (slots={args.num_slots or 'auto'}), COLD + boosted clock"
    else:
        timer = f"EAGER + L2 flush ({args.flush_mb} MiB), COLD cache (HBM-bound)"
    print("=" * 72)
    print("  a8w4/mxscale TDM-load ablation on gfx1250 (one process per variant)")
    print(f"  Shape M={args.M} N={args.N} K={args.K}  "
          f"tile=({args.tile_m},{args.tile_n},{args.tile_k}) "
          f"warps=({args.m_warp}x{args.n_warp}) nb={args.num_buffers} b_layout={args.b_layout}")
    print(f"  Timer: {timer} (warmup={args.warmup}, iters={args.iters})")
    print("=" * 72)

    rows = []
    for n in names:
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--single", n] + shape_args,
            capture_output=True, text=True,
        )
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT,")), None)
        if line is None:
            print(f"  {n:9s}  FAILED\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
            continue
        _, name, loaded, us, cs = line.split(",")
        us = float(us)
        rows.append((name, loaded, us, float(cs)))
        print(f"  {name:9s} [{loaded:20s}]  {us:7.3f} us   checksum={float(cs):.3e}")

    real_us = next((u for nm, _, u, _ in rows if nm == "real"), None)
    print("=" * 72)
    print(f"  {'variant':10s} {'us':>8s}   {'Δ vs none':>10s}   {'% of real':>9s}")
    none_us = next((u for nm, _, u, _ in rows if nm == "none"), None)
    for name, loaded, us, _ in rows:
        d = f"{us - none_us:+7.3f}" if none_us is not None else ""
        frac = f"{100.0 * us / real_us:6.1f}%" if real_us else ""
        print(f"  {name:10s} {us:8.3f}   {d:>10s}   {frac:>9s}")
    print("=" * 72)

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("variant,loaded,us,checksum\n")
            for name, loaded, us, cs in rows:
                f.write(f"{name},{loaded},{us:.4f},{cs:.6e}\n")
        print(f"  wrote {args.csv}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-M", type=int, default=16)
    p.add_argument("-N", type=int, default=2048 * 2 * 256)
    p.add_argument("-K", type=int, default=4096)
    p.add_argument("--tile-m", type=int, default=16)
    p.add_argument("--tile-n", type=int, default=256)
    p.add_argument("--tile-k", type=int, default=256)
    p.add_argument("--m-warp", type=int, default=1)
    p.add_argument("--n-warp", type=int, default=4)
    p.add_argument("--num-buffers", type=int, default=4)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--flush-mb", type=int, default=512,
                   help="L2/MALL flush scratch size (MiB) cleared before each timed launch")
    p.add_argument("--method", type=str, default="eager", choices=["eager", "graph"],
                   help="eager = per-iter L2 flush (real single-launch cost); graph = rotating-buffer cold + boosted clock")
    p.add_argument("--num-slots", type=int, default=0, help="graph rotation depth (0=auto, 1=hot single)")
    p.add_argument("--b-layout", type=str, default="preshuffle", choices=["preshuffle", "tiled"],
                   help="B preshuffle layout: 'preshuffle' = dim0=packed_tile_k_b*16 (wide rows); "
                        "'tiled' = dim0=256 (flat block array via preshuffle_b_16x16_tiled)")
    p.add_argument("--variants", type=str, default="none,a,b0,b1,b,asv,b_asv,bs,scale_prod,ab,all,all_prod",
                   help="comma list from: " + ",".join(VARIANTS))
    p.add_argument("--with-real-gemm", action="store_true", default=False,
                   help="also time the production a8w4 mxscale GEMM under the same timer")
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--single", type=str, default=None,
                   help="internal: measure one variant (or 'real') in this process")
    args = p.parse_args()

    if args.single is not None:
        _measure_single(args.single, args)
    else:
        _sweep(args)


if __name__ == "__main__":
    main()

