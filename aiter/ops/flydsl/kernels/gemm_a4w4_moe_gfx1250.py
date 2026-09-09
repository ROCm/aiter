# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Grouped contiguous-M A4W4 MoE GEMM for gfx1250 (quadrant pipeline, 4x4 cluster).

The K-loop is FlyDSL's dense ``kernels/gemm/gemm_a4w4_256x256_gfx1250.py``
pipeline, taken over unchanged in shape: per-wave TDM ownership (one wave each
for A / B / A-scale / B-scale), a planar operand-major LDS arena, four 64x64
WMMA quadrants per K-tile with one DS producer group hosted inside each, the
``_DsOrder`` dscnt ledger, the split READY/REUSE fence, and the two wave-parity
schedules.

Grafted on top is the MoE plumbing from ``mxfp4_preshuffle_gfx1250_tdm.py``:
the contiguous-M expert bisect, per-expert B / B-scale batch offsets, the
preshuffled A-scale layout aiter's quantizer emits, and both epilogues (fused
activation + MXFP4 quant for gemm1, bf16/f16 passthrough for gemm2).

Cluster contract (differs from the 1-D ``cluster_n`` kernel it replaces)
-----------------------------------------------------------------------
The launch is a ``(cluster_m, cluster_n, 1)`` workgroup cluster over
``(grid.x = M tiles, grid.y = N tiles)``.  A and A-scale are multicast down a
cluster column (peers share the M tile), B and B-scale across a cluster row
(peers share the N tile).  B is *per expert*, so the ``cluster_m`` M-tiles a
cluster row spans MUST belong to one expert -- otherwise a peer receives another
expert's weights.  That is why the caller aligns each expert's contiguous-M
block to ``tile_m * cluster_m`` rows rather than ``tile_m``; see
``grouped_moe_gfx1250._contiguous_align``.  The same alignment is what makes the
``expert < n_experts`` skip cluster-uniform, so peers all run or all skip and
the pairwise-matched multicast loads never deadlock.
"""

import glob
import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.rocdl import cluster, tdm_ops
from flydsl.expr.typing import Constexpr, T, as_ir_value
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import check_smem_capacity

from aiter.utility.mx_types import MxDtypeInt as MxDtype

from .gemm_common_gfx1250 import (
    LOG2E,
    lds_store_b16_raw,
    make_sgpr_opaque,
    make_vgpr_opaque,
    batched_silu_swiglu,
    batched_situv2,
    fused_silu_swiglu_elem,
    fclamp_f32,
    fmin_f32,
    fused_situv2_elem,
    make_lds_copy_ops,
    pipeline_fence,
    pipeline_fence_signal,
    pipeline_fence_wait,
    situv2_consts,
    workgroup_barrier,
)
from .quant_utils import (
    emit_amax_e8m0_native_scale,
    emit_cvt_scalef32_pk8_fp4_bf16,
)
from .tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE,
)

# The profile the ported pipeline is written against. It is not a tuning knob:
# the quadrant plan, the seed banks and the planar LDS arena are all sized by it.
SUPPORTED_PROFILE = (256, 256, 256, 2, 2, 4)
SUPPORTED_CLUSTER = (2, 2)

# PERF PROBE ONLY -- a bitmask of epilogue ablations. Every non-zero value
# produces numerically WRONG output; must stay 0 outside a probe. Each bit
# replaces one instruction class with something cheaper that yields the SAME
# number of live values, so nothing downstream is dead-code-eliminated and the
# A/B attributes only that class. Bits compose.
#   1  drop the gpt-oss clamp (2 v_med3 per output value)
#   2  drop the sigmoid (mul 0.5 + v_tanh + v_fma); keep clamp and the *u mul
#   4  drop the amax / e8m0 block scale (constant scale instead)
#   8  drop the cross-lane peer fetch (v_permlanex16, duplicate own instead)
_EPI_PROBE = 0

# Exchange the kgrp peer's half of each pk8 group as 2 packed bf16 dwords
# instead of 4 loose f32 lanes -- see the comment at the pack site.
_EPI_PACK_BF16_XCHG = True
# Clamp the up operand at the final multiply instead of alongside the gate, so
# it is not live across the TRANS stage.
_EPI_LATE_U_CLAMP = False
# Hold the clamp bounds in VGPRs rather than SGPRs/literals: a VOP3 v_med3 that
# reads two different SGPRs pays for the narrow scalar operand port.
_EPI_CLAMP_VGPR = True
# Store each half-wave's own 2 bytes of fp4 instead of assembling a 4-byte dword
# across the kgrp pair -- removes the pack's cross-lane shuffles entirely.
_EPI_PACK_B16_STORE = False

# Epilogue activation scheduling. All numerics-preserving: the activation is
# elementwise, so batch width only changes instruction scheduling.
#   _EPI_ACT_BLKS    -- MX blocks activated per batch (16 values per lane each).
#                       Must divide N_MX_BLKS.
#   _EPI_ACT_BARRIERS -- keep the sched_barrier walls between the activation's
#                       clamp / tanh / fma / mul stages. Measured inside the
#                       noise either way (649.8 off vs 653.2 on).
#
# The best width depends on how long the TRANS dependency chain is, and it
# INVERTS between the two formulations (gemm1 us, --iters 128, const-init):
#
#              16 vals   32 vals   64 vals
#   exp2+rcp   1044.7     796.8      --
#   v_tanh      645.6     655.1     654.5
#
# With two serial TRANS ops per element there is nothing to cover the latency --
# this kernel is 1008 VGPRs, one wave per SIMD, so no other wave is resident --
# and narrow batching is catastrophic. With one TRANS op the chain is short
# enough that fewer live values beats more ILP, and >= 32 is flat. Do not
# re-tune one of these knobs without re-checking the other.
_EPI_ACT_BLKS = 1
_EPI_ACT_BARRIERS = True
# Evaluate g*sigmoid(g) as m*(1+tanh(m)) on gfx1250's v_tanh_f32 instead of the
# exp2 -> add -> rcp chain: one TRANS op per element instead of two.
_EPI_HW_TANH = True
# 16-row blocks whose activation is issued as one batch. Batch width is the
# dominant epilogue knob (see the comment at the loop) -- it must divide
# wmma_m_rep (8 here).
_EPI_ACT_WM = 1


def supports(tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers):
    """Whether this kernel can serve the given tile profile."""
    return (
        int(tile_m),
        int(tile_n),
        int(tile_k),
        int(m_warp),
        int(n_warp),
        int(num_buffers),
    ) == SUPPORTED_PROFILE


def _mcast_masks(local_x, local_y, cluster_m, cluster_n):
    """MCAST workgroup masks for A (share M tile) and B (share N tile).

    Flat WG index inside a cluster is X-inner (``local_x + local_y*cluster_m``),
    so the A group varies ``local_y`` at fixed ``local_x`` and the B group
    varies ``local_x`` at fixed ``local_y``. Local copy of FlyDSL's
    ``kernels.common.gfx1250_cluster.compute_mcast_masks``; that package is not
    importable from aiter.
    """
    lx = arith.index_cast(T.i32, local_x)
    ly = arith.index_cast(T.i32, local_y)
    a_pattern = 0
    for ly_i in range(cluster_n):
        a_pattern |= 1 << (ly_i * cluster_m)
    a_mask = arith.shli(arith.constant(a_pattern, type=T.i32), lx)
    b_pattern = arith.constant((1 << cluster_m) - 1, type=T.i32)
    col_base = arith.muli(ly, arith.constant(cluster_m, type=T.i32))
    b_mask = arith.shli(b_pattern, col_base)
    return a_mask, b_mask


# Largest cluster the mcast masks can address: they are 32-bit workgroup masks.
MCAST_MASK_WGS = 32
# Used when the KFD topology cannot be read. gfx1250 is 2 shader arrays per
# shader engine, and this kernel takes a whole CU's LDS, so one workgroup per CU.
FALLBACK_CUS_PER_SE = 16

_KFD_NODES = "/sys/class/kfd/kfd/topology/nodes"


def _gpu_topology():
    """``(enabled CUs per shader engine, LDS bytes per CU)`` from KFD.

    ``None`` when the topology is unreadable, which leaves the caller on its
    conservative built-in bound rather than guessing upward.
    """
    for path in sorted(glob.glob(f"{_KFD_NODES}/*/properties")):
        props = {}
        try:
            with open(path) as f:
                for line in f:
                    key, _, value = line.partition(" ")
                    try:
                        props[key] = int(value)
                    except ValueError:
                        pass
        except OSError:
            continue
        simds = props.get("simd_count", 0)
        simd_per_cu = props.get("simd_per_cu", 0)
        arrays = props.get("array_count", 0)
        arrays_per_engine = props.get("simd_arrays_per_engine", 0)
        lds_kb = props.get("lds_size_in_kb", 0)
        if not (simds and simd_per_cu and arrays and arrays_per_engine and lds_kb):
            continue  # a CPU node, or a kernel that does not report these
        engines = arrays // arrays_per_engine
        cus = simds // simd_per_cu
        if engines < 1 or cus < engines:
            continue
        return cus // engines, lds_kb * 1024
    return None


def max_cluster_workgroups(lds_bytes_per_wg):
    """How many workgroups of this size can share one cluster.

    A cluster's workgroups multicast into each other's LDS and meet at
    ``cluster_barrier``, so every member must be RESIDENT AT THE SAME TIME. A
    cluster the hardware cannot place does not fail -- it hangs. The bound is
    therefore a co-residency bound: a cluster is placed inside one shader
    engine, so it cannot exceed that engine's workgroup slots,

        (enabled CUs per SE) * (LDS per CU // LDS per workgroup)

    which on this part (256 CUs / 16 SEs, 320 KiB per CU) is 16 for a kernel
    whose arena is a whole CU's LDS.

    The HIP runtime is NOT a sufficient gate. ``hipOccupancyMaxPotentialCluster
    Size`` answers 18 -- 2 shader arrays x 9 PHYSICAL CUs -- and ``hipOccupancy
    MaxActiveClusters`` likewise accepts 17 and 18; two CUs per SE are harvested
    on this part, so those clusters pass every runtime check and then hang.
    Counting *enabled* CUs is what makes the bound sound.
    """
    topology = _gpu_topology()
    if topology is None:
        return min(MCAST_MASK_WGS, FALLBACK_CUS_PER_SE)
    cus_per_se, lds_per_cu = topology
    wgs_per_cu = max(1, lds_per_cu // max(1, int(lds_bytes_per_wg)))
    return max(1, min(MCAST_MASK_WGS, cus_per_se * wgs_per_cu))


def num_xcds():
    """Shader engines the dispatcher round-robins workgroups over (``num_xcc``).

    Read from KFD rather than hardcoded: the swizzle below *inverts* this
    round-robin, so a wrong count does not merely fail to help, it scatters
    tiles that should share an L2. Falls back to the 8 this part reports.
    """
    for path in sorted(glob.glob(f"{_KFD_NODES}/*/properties")):
        try:
            with open(path) as f:
                for line in f:
                    key, _, value = line.partition(" ")
                    if key == "num_xcc":
                        n = int(value)
                        if n >= 1:
                            return n
        except (OSError, ValueError):
            continue
    return 8


def _swizzled_blk(bid_x, bid_y, bid_z, cluster_m, cluster_n, tile_m, tile_n, wgm):
    """``(blk_m, blk_n)`` under the XCD-aware cluster order.

    A function, and called from a TERNARY rather than an ``if`` statement: the
    FlyDSL rewriter turns an ``if`` into a traced branch, and names bound inside
    one do not escape it -- which shows up as a NameError on the *next* use, not
    at the branch. mxfp4_preshuffle_gfx1250_tdm carries the same warning.
    """
    cl_x = bid_x // cluster_m
    cl_y = bid_y // cluster_n
    # Intra-cluster position -- the pair compute_cluster_position returns.
    # Carried through untouched so every peer keeps the M tile its multicast
    # mask assumes.
    lx = bid_x - cl_x * cluster_m
    ly = bid_y - cl_y * cluster_n
    cl_per_run = fx.grid_dim.x // cluster_m
    num_cl_n = fx.grid_dim.y // cluster_n
    # Linear cluster id in dispatch order: x fastest, then y, then z.
    cid = (bid_z * num_cl_n + cl_y) * cl_per_run + cl_x
    m_cl, n_cl = _xcd_cluster_swizzle(
        cid, cl_per_run * fx.grid_dim.z, num_cl_n, wgm, num_xcds()
    )
    return (m_cl * cluster_m + lx) * tile_m, (n_cl * cluster_n + ly) * tile_n


def _xcd_cluster_swizzle(cid, num_cl_m, num_cl_n, wgm, n_xcds):
    """Remap a linear CLUSTER id to ``(m_cluster, n_cluster)`` for L2 reuse.

    Cluster-granular on purpose: a cluster is never split, so peers keep sharing
    one M tile, which is what the A multicast and the cluster-uniform expert
    bisect both rely on. Splitting a cluster here would not fail loudly -- it
    would multicast the wrong rows.

    Two effects, in order, following ``gemm_a8w8_8wave._xcd_swizzle_any``:
    1. Invert the dispatcher's round-robin (cluster i -> XCD ``i % n_xcds``) so
       that clusters sharing an XCD get CONSECUTIVE ids. Without this the
       ``tile_m``-adjacent clusters that read the same expert's B block land on
       different XCDs and each pulls its own copy through its own L2.
    2. Group ``wgm`` M-clusters and sweep all N inside the group, so that group's
       A/B stay resident across the sweep.
    """
    num_cl = num_cl_m * num_cl_n
    xcd = cid % n_xcds
    intra = cid // n_xcds
    base = num_cl // n_xcds
    extra = num_cl - base * n_xcds
    cid2 = xcd * base + (xcd < extra).select(xcd, extra) + intra

    span = wgm * num_cl_n
    grp = cid2 // span
    intra_grp = cid2 - grp * span
    first_m = grp * wgm
    rem = num_cl_m - first_m
    gsz = (rem < wgm).select(rem, fx.Int32(wgm))
    n_cl = intra_grp // gsz
    m_cl = first_m + (intra_grp - n_cl * gsz)
    return m_cl, n_cl


@flyc.jit
def launch_gemm_a4w4_moe(
    arg_c: fx.Tensor,
    arg_a: fx.Pointer,
    arg_b: fx.Pointer,
    arg_scale_a: fx.Tensor,
    arg_scale_b: fx.Tensor,
    arg_m_tile_map: fx.Pointer,
    arg_bias: fx.Pointer,
    arg_quant_scale: fx.Tensor,
    i32_m: fx.Int32,
    stream: fx.Stream,
    N: fx.Int32,
    K: Constexpr[int],
    tile_m: Constexpr[int],
    tile_n: Constexpr[int],
    tile_k: Constexpr[int],
    m_warp: Constexpr[int],
    n_warp: Constexpr[int],
    out_is_f16: Constexpr[int],
    num_buffers: Constexpr[int],
    n_experts: Constexpr[int],
    stage1_act: Constexpr[int],
    has_bias: Constexpr[int],
    f32_swiglu_limit: fx.Float32,
    stage1_quant_out: Constexpr[int] = 0,
    quant_wmma_rep: Constexpr[int] = 1,
    cluster_m: Constexpr[int] = 4,
    cluster_n: Constexpr[int] = 4,
    f32_situ_beta: fx.Float32 = 1.0,
    f32_situ_linear_beta: fx.Float32 = 1.0,
    act_has_limit: Constexpr[int] = 1,
    xcd_swizzle: Constexpr[int] = 0,
):
    """Launch the grouped contiguous-M a4w4 MoE GEMM on the quadrant pipeline."""
    assert supports(
        tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers
    ), f"only the tuned {SUPPORTED_PROFILE} profile is supported"
    # 4x4 is the tuned cluster; the pipeline is parameterized on any
    # representable shape for sweeps. Representability is only the cheap half of
    # the contract -- the co-residency half needs ARENA_B and is asserted below.
    assert cluster_m >= 1 and cluster_n >= 1, (
        f"cluster {cluster_m}x{cluster_n} must be positive"
    )
    assert cluster_m * cluster_n <= MCAST_MASK_WGS, (
        f"cluster {cluster_m}x{cluster_n} exceeds the {MCAST_MASK_WGS}-bit mcast "
        "workgroup mask"
    )
    assert K % tile_k == 0 and K // tile_k > num_buffers, (
        f"K={K} needs to be a multiple of tile_k={tile_k} with more than "
        f"{num_buffers} K-tiles"
    )
    # 0 = off (the raw row-major map). >0 selects the XCD-aware tile order and
    # is the group width in M-clusters, matching the repo's xcd_swizzle
    # convention (gemm_a8w8_8wave, moe_kernels) where the value IS the wgm.
    assert xcd_swizzle >= 0, f"xcd_swizzle={xcd_swizzle} must be >= 0"

    cluster_sync_revs = 8
    m_run_max, m_run_min = 32, 8
    FENCE_WAIT_POS = 7  # WMMAs of slack between the READY signal and its wait
    WMMA_M = 32  # N-major operand (the weights)
    WMMA_N = 16
    WMMA_K = 128
    WAVE = fx.num_warp_threads()
    PACK_TK = tile_k // 2  # A/B row bytes per K-tile (FP4 packed 2/byte)
    K_STEPS = tile_k // WMMA_K
    SC_WORDS = tile_k // 4  # B-scale i32 words per super-row per K-tile
    SB_SUPERS = tile_n // 32
    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp
    wmma_m_rep = warp_tile_m // WMMA_N  # A fragments (16 rows each)
    wmma_n_rep = warp_tile_n // WMMA_M  # B fragments (32 cols each)
    half_m, half_n = wmma_m_rep // 2, wmma_n_rep // 2
    WMMA_PER_Q = half_m * half_n * K_STEPS  # WMMAs in one 64x64 quadrant
    n_acc = wmma_m_rep * wmma_n_rep
    output_n_rep = warp_tile_n // 16  # 16-col output fragments per wave row
    num_waves = m_warp * n_warp
    block = num_waves * WAVE

    LDS_PAD_A = 16
    A_LDS_ROW = PACK_TK + LDS_PAD_A
    B_LDS_ROW = PACK_TK * 16
    # A-scale rides aiter's preshuffled e8m0 layout, not the dense kernel's
    # (M/32, K) one: [M/128][k128][wm][lane16] dwords, i.e. one dword per
    # (row, k128) with the WMMA operand already contiguous. Both give the
    # 32-lane-per-super register the atom wants, so only the strides change.
    AS_GROUP_ROWS = wmma_m_rep * 16  # M rows one preshuffled outer row covers
    AS_SUPERS = tile_m // AS_GROUP_ROWS  # == m_warp
    AS_INNER_B = (tile_k // WMMA_K) * wmma_m_rep * 16 * 4  # bytes per K-tile
    AS_ROW_B = (K // WMMA_K) * wmma_m_rep * 16 * 4  # global bytes per outer row

    STAGE_A = tile_m * A_LDS_ROW
    STAGE_B = (tile_n // 16) * B_LDS_ROW
    STAGE_SA = AS_SUPERS * AS_INNER_B
    STAGE_SB = SB_SUPERS * tile_k
    # Operand-major planar LDS: scales and A below, B on the next 64-KiB
    # boundary above them.
    PLANAR_SA_BASE = 0
    PLANAR_A_BASE = PLANAR_SA_BASE + num_buffers * STAGE_SA
    PLANAR_SB_BASE = PLANAR_A_BASE + num_buffers * STAGE_A
    PLANAR_B_BASE = ((PLANAR_SB_BASE + num_buffers * STAGE_SB + 65535) // 65536) * 65536
    PLANAR_END = PLANAR_B_BASE + num_buffers * STAGE_B

    # This kernel is a4w4-only, so a quantized stage-1 output is always MXFP4.
    is_fp4_quant = bool(stage1_quant_out and stage1_act)
    if stage1_quant_out:
        assert stage1_act, "stage1_quant_out is only defined with an activation"
    # Each wn subtile yields 8 activated cols (4 per kgrp); 4 subtiles = 32 cols
    # = one MX block.
    WN_PER_MX_BLOCK = 4
    if is_fp4_quant:
        assert (
            output_n_rep % WN_PER_MX_BLOCK == 0
        ), "stage1 quant requires complete four-WMMA N groups"
    # Output row pitch in LDS. An activation fuses gate+up, halving the column
    # count; the fp4 quant payload halves it again (2 values per byte). The
    # passthrough tile pads +16 cols so a b128 store spreads off one bank.
    STORE_N = tile_n // (4 if is_fp4_quant else 2) if stage1_act else tile_n
    STORE_PAD = 0 if stage1_act else 16
    STORE_PITCH = STORE_N + STORE_PAD
    C_STORE_B = tile_m * STORE_PITCH * (1 if is_fp4_quant else 2)

    ARENA_B = max(PLANAR_END, C_STORE_B)
    check_smem_capacity(ARENA_B, str(get_hip_arch()))
    # Reject a cluster the hardware cannot co-schedule, here, rather than
    # deadlocking the device with it -- see max_cluster_workgroups.
    _cluster_cap = max_cluster_workgroups(ARENA_B)
    assert cluster_m * cluster_n <= _cluster_cap, (
        f"cluster {cluster_m}x{cluster_n} = {cluster_m * cluster_n} workgroups "
        f"cannot be co-resident: at {ARENA_B} B of LDS each, one shader engine "
        f"holds {_cluster_cap}. Such a cluster hangs instead of failing, so it "
        "is rejected before launch."
    )

    _act = f"_act{stage1_act}" if stage1_act else ""
    _qout = f"_q{stage1_quant_out}r{quant_wmma_rep}" if stage1_quant_out else ""
    _bias = "_bias" if has_bias else ""
    _probe = f"_probe{_EPI_PROBE}" if _EPI_PROBE else ""
    _epi = f"_eb{_EPI_ACT_BLKS}{'' if _EPI_ACT_BARRIERS else 'nb'}"
    _epi += "_th" if _EPI_HW_TANH else ""
    _epi += f"_wm{_EPI_ACT_WM}"
    _epi += "_bx" if _EPI_PACK_BF16_XCHG else ""
    _epi += "_lu" if _EPI_LATE_U_CLAMP else ""
    _epi += "_cv2" if _EPI_CLAMP_VGPR else ""
    _epi += "_b16" if _EPI_PACK_B16_STORE else ""
    _epi += "" if act_has_limit else "_nolim"
    _kname = (
        f"a4w4_quad_t{tile_m}x{tile_n}x{tile_k}_w{m_warp}x{n_warp}"
        f"_b{num_buffers}_K{K}_e{n_experts}"
        f"{_act}{_bias}{_qout}_cl{cluster_m}x{cluster_n}{_probe}{_epi}"
        f"{f'_xcd{xcd_swizzle}' if xcd_swizzle else ''}"
    )

    @flyc.kernel(name=_kname, known_block_size=[block, 1, 1])
    def kernel(
        arg_c: fx.Tensor,
        arg_a: fx.Pointer,
        arg_b: fx.Pointer,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        arg_m_tile_map: fx.Pointer,
        arg_bias: fx.Pointer,
        arg_quant_scale: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        i32_k: fx.Int32,
        f32_swiglu_limit: fx.Float32,
        f32_situ_beta: fx.Float32,
        f32_situ_linear_beta: fx.Float32,
    ):
        # K_TILES stays runtime even though K is a Constexpr: a Python int here
        # would unroll the revolution loop into ~1.5k WMMAs of I-cache.
        K_TILES = i32_k // tile_k
        Kp16 = (K // 2) * 16  # B bytes per 16-row outer block, whole K
        K4 = K // 4  # B-scale dwords per super row, whole K

        tid = fx.thread_idx.x
        bid_x, bid_y, bid_z = fx.block_idx
        wave = fx.Int32(rocdl.readfirstlane(T.i32, tid // WAVE))
        lane = tid % WAVE
        lane16 = lane % 16
        kgrp = lane // 16
        wave_m = wave // n_warp
        wave_n = wave % n_warp
        local_x, local_y = cluster.compute_cluster_position()
        a_mask, b_mask = _mcast_masks(local_x, local_y, cluster_m, cluster_n)

        m_chunk = bid_z
        # Ternary, never an `if` statement -- see _swizzled_blk. xcd_swizzle is a
        # Constexpr, so only the taken side is ever built.
        blk_m, blk_n = (
            _swizzled_blk(
                bid_x, bid_y, bid_z, cluster_m, cluster_n, tile_m, tile_n, xcd_swizzle
            )
            if xcd_swizzle
            else ((m_chunk * fx.grid_dim.x + bid_x) * tile_m, bid_y * tile_n)
        )
        blk_m64 = fx.Int64(blk_m)
        blk_n64 = fx.Int64(blk_n)

        # In-kernel bisect: the expert owning this M tile, from the psum map.
        # Cluster-uniform because a cluster's cluster_m M-tiles are inside one
        # expert by the caller's tile_m*cluster_m alignment.
        i32_ptr = fx.PointerType.get(
            elem_ty=fx.Int32.ir_type, address_space=fx.AddressSpace.Global, alignment=4
        )
        tile_map = fx.recast_iter(i32_ptr, arg_m_tile_map)
        lo, hi = blk_m * 0, blk_m * 0 + n_experts
        for _ in range_constexpr(max(1, math.ceil(math.log2(max(2, n_experts))) + 1)):
            mid = (lo + hi) >> 1
            mid_clamped = (mid < n_experts - 1).select(mid, n_experts - 1)
            go_right = tile_map[mid_clamped] <= blk_m
            lo = go_right.select(mid + 1, lo)
            hi = go_right.select(hi, mid)
        expert = lo
        eb64 = fx.Int64(expert)
        # Per-expert A / C row bound: rows past this expert's block are padding.
        mn_oob = tile_map[(expert < n_experts).select(expert, n_experts - 1)] - blk_m

        arena = fx.SharedAllocator(static=False)
        arena.allocate(ARENA_B)
        base_ptr = arena.base_ptr

        def _planar_base(offset, stride, stage):
            ptr = fx.add_offset(base_ptr, offset + stage * stride)
            return fx.Index(fx.ptrtoint(ptr))

        def _view(ptr, shape, stride):
            return fx.Tensor(fx.make_view(ptr, fx.make_layout(shape, stride)))

        def _gv(base, off, shape, stride):
            return _view(fx.add_offset(base, off), shape, stride)

        oc = fx.Float16 if out_is_f16 else fx.BFloat16
        out_elem = T.f16 if out_is_f16 else T.bf16

        # Skip padding tiles (expert id == n_experts). Uniform across the whole
        # cluster, so the pairwise-matched multicast loads stay balanced.
        if expert < n_experts:
            n64 = fx.Int64(i32_n)
            B_BATCH_ROWS = n64 // 16
            N_SUPERS = (n64 + 31) // 32

            gA_base = fx.recast_iter(fx.Int8, arg_a)
            gB_base = fx.recast_iter(fx.Int8, arg_b)
            gSA_base = fx.recast_iter(fx.Int8, fx.get_iter(arg_scale_a))
            gSB_base = fx.recast_iter(fx.Int8, fx.get_iter(arg_scale_b))

            a_off0 = blk_m64 * fx.Int64(K // 2)
            b_off0 = (eb64 * B_BATCH_ROWS + blk_n64 // 16) * fx.Int64(Kp16)
            sa_off0 = (blk_m64 // AS_GROUP_ROWS) * fx.Int64(AS_ROW_B)
            sb_off0 = (eb64 * N_SUPERS + blk_n64 // 32) * fx.Int64(K4 * 4)

            gA = _gv(gA_base, a_off0, (tile_m, PACK_TK), (PACK_TK, 1))
            gB = _gv(gB_base, b_off0, (tile_n // 16, B_LDS_ROW), (B_LDS_ROW, 1))
            gSA = _gv(gSA_base, sa_off0, (AS_SUPERS, AS_INNER_B), (AS_INNER_B, 1))
            gSB = _gv(gSB_base, sb_off0, (SB_SUPERS, tile_k), (tile_k, 1))

            def _build_tdm_desc(owner):
                if const_expr(owner == 0):
                    tensor, offset = gA, PLANAR_A_BASE
                    shape, lds_stride = (tile_m, PACK_TK), A_LDS_ROW
                    stride, mask = K // 2, a_mask
                    bound, pad, early = mn_oob, LDS_PAD_A, False
                elif const_expr(owner == 1):
                    tensor, offset = gB, PLANAR_B_BASE
                    shape, lds_stride = (tile_n // 16, B_LDS_ROW), B_LDS_ROW
                    stride, mask = Kp16, b_mask
                    bound, pad, early = None, 0, False
                elif const_expr(owner == 2):
                    tensor, offset = gSA, PLANAR_SA_BASE
                    shape, lds_stride = (AS_SUPERS, AS_INNER_B), AS_INNER_B
                    stride, mask = AS_ROW_B, a_mask
                    bound, pad, early = None, 0, True
                else:
                    tensor, offset = gSB, PLANAR_SB_BASE
                    shape, lds_stride = (SB_SUPERS, tile_k), tile_k
                    stride, mask = K, b_mask
                    bound, pad, early = None, 0, True
                desc = tdm_ops.make_tensor_descriptor_2d(
                    global_ptr=tensor,
                    lds_memref=_view(
                        fx.add_offset(base_ptr, offset), shape, (lds_stride, 1)
                    ),
                    global_offset=(0, 0),
                    tensor_shape=shape,
                    strides=(stride, 1),
                    tile_shape=shape,
                    elem_bytes=1,
                    pad_interval=shape[1] if pad else 0,
                    pad_amount=pad,
                    num_warps=1,
                    workgroup_mask=mask,
                    early_timeout=early,
                    oob_outer_bound=bound,
                )
                return desc, shape[0] * lds_stride, shape[1]

            def _owned_tdm_desc(owner):
                desc, lds_step, global_step = _build_tdm_desc(owner)
                return (
                    Vec(desc.dgroup0),
                    Vec(desc.dgroup1),
                    fx.Int32(lds_step),
                    fx.Int32(global_step),
                )

            dgroup0 = Vec.from_elements([fx.Int32(0)] * 4, fx.Int32)
            dgroup1 = Vec.from_elements([fx.Int32(0)] * 8, fx.Int32)
            tdm_lds_step, tdm_global_step = fx.Int32(0), fx.Int32(0)
            if wave == 0:
                dgroup0, dgroup1, tdm_lds_step, tdm_global_step = _owned_tdm_desc(0)
            elif wave == 1:
                dgroup0, dgroup1, tdm_lds_step, tdm_global_step = _owned_tdm_desc(1)
            elif wave == 2:
                dgroup0, dgroup1, tdm_lds_step, tdm_global_step = _owned_tdm_desc(2)
            else:
                dgroup0, dgroup1, tdm_lds_step, tdm_global_step = _owned_tdm_desc(3)

            tdm_desc = tdm_ops.TDMDescriptor2D(as_ir_value(dgroup0), as_ir_value(dgroup1))
            tdm_base_lds, tdm_base_lo, tdm_base_hi = dgroup0[1], dgroup0[2], dgroup0[3]

            def _prepare_tdm(slot, tile):
                desc = tdm_ops.update_tensor_descriptor_2d_lds_addr(
                    tdm_desc, tdm_base_lds + tdm_lds_step * fx.Int32(slot)
                )
                return tdm_ops.update_tensor_descriptor_2d_addr64(
                    desc,
                    tdm_base_lo,
                    tdm_base_hi,
                    fx.Int32(tile) * tdm_global_step,
                )

            wmb = wave_m * warp_tile_m
            wnb = wave_n * warp_tile_n

            # opsel_b selects which half of the shared 32-lane A-scale word this
            # 16-row block uses.
            wmma_atoms = [
                fx.make_mma_atom(
                    fx.rocdl.WMMAScale(
                        WMMA_M,
                        WMMA_N,
                        WMMA_K,
                        fx.Float4E2M1FN,
                        fx.Float4E2M1FN,
                        fx.Float32,
                        opsel_b=sa_sel,
                    )
                )
                for sa_sel in range_constexpr(2)
            ]
            c_frags = [fx.make_rmem_tensor(16, fx.Float32) for _ in range_constexpr(n_acc)]
            for cf in c_frags:
                cf.store(Vec.filled(16, 0.0, fx.Float32))

            def _rmem(n, v):
                t = fx.make_rmem_tensor(n, fx.Int32)
                t.store(v)
                return t

            def _mma_one(wm, wn, k, act, wt, sa_k, sb_k):
                idx = wm * wmma_n_rep + wn
                fx.gemm(
                    wmma_atoms[wm % 2],
                    c_frags[idx],
                    wt,
                    act,
                    c_frags[idx],
                    scale_a=sb_k[wn * K_STEPS + k],
                    scale_b=sa_k[(wm // 2) * K_STEPS + k],
                )

            def _mma_block_range(
                wm0, wn0, act, wt, sa_k, sb_k, start, count, n_index_fast=False
            ):
                """Issue ``count`` WMMAs from linear position ``start``."""
                n_m, n_n = len(act), len(wt)
                n_minor = n_n if n_index_fast else n_m
                for linear in range_constexpr(count):
                    pos = start + linear
                    minor = pos % n_minor
                    k = (pos // n_minor) % K_STEPS
                    major = pos // (n_minor * K_STEPS)
                    i, j = (major, minor) if n_index_fast else (minor, major)
                    _mma_one(wm0 + i, wn0 + j, k, act[i][k], wt[j][k], sa_k, sb_k)

            cluster.cluster_barrier()
            # Keep fragment displacements as DS immediates inside the K loop.
            stage_a_addr, stage_b_addr, stage_sa_addr, stage_sb_addr = [], [], [], []
            sb_col = wnb + lane
            a_byte = fx.Index((wmb + lane16) * A_LDS_ROW + kgrp * 16)
            a_frag_row_step = 16 * A_LDS_ROW
            b_byte = fx.Index((wnb // 16) * B_LDS_ROW + kgrp * 256 + lane16 * 16)
            # Preshuffled A-scale: lane spans the 32 rows of a wm pair, and the
            # atom's opsel_b picks the 16-row half.
            sa_byte = fx.Index(wave_m * AS_INNER_B + lane * 4)
            sb_byte = fx.Index(((sb_col // 32) * SC_WORDS + sb_col % 32) * 4)
            for addr_stage in range_constexpr(num_buffers):
                stage_a_addr.append(_planar_base(PLANAR_A_BASE, STAGE_A, addr_stage) + a_byte)
                stage_b_addr.append(_planar_base(PLANAR_B_BASE, STAGE_B, addr_stage) + b_byte)
                stage_sa_addr.append(
                    _planar_base(PLANAR_SA_BASE, STAGE_SA, addr_stage) + sa_byte
                )
                stage_sb_addr.append(
                    _planar_base(PLANAR_SB_BASE, STAGE_SB, addr_stage) + sb_byte
                )

            lds_load_b32, _ = make_lds_copy_ops(32)
            lds_load_b128, lds_store_b128 = make_lds_copy_ops(128)
            _, lds_store_b32 = make_lds_copy_ops(32)
            _, lds_store_b64 = make_lds_copy_ops(64)

            class _DsOrder:

                LIMIT = 63

                def __init__(self):
                    self.issued = 0

                def mark(self, ops):
                    self.issued += ops
                    return self.issued

                def wait(self, mark):
                    rocdl.s_wait_dscnt(min(self.issued - mark, self.LIMIT))

            ds = _DsOrder()

            DS_A_OPS, DS_B_OPS = 2, 4  # ds_load_b128 per A / B fragment

            def _stage_load_a(stage, wm, k):
                """16 rows x 128 FP4: lane holds row wm*16+lane16, K-bytes kgrp*16+32j."""
                row_off = wm * a_frag_row_step + k * (WMMA_K // 2)
                v0 = lds_load_b128(stage_a_addr[stage], row_off)
                v1 = lds_load_b128(stage_a_addr[stage], row_off + 32)
                return v0.shuffle(v1, list(range(8))), ds.mark(DS_A_OPS)

            def _stage_load_b(stage, wn, k):
                """32 cols x 128 FP4: two stacked 16-col blocks, low 8 words then high."""
                col_off = wn * 2 * B_LDS_ROW + k * (WMMA_K // 2) * 16
                v = [
                    lds_load_b128(stage_b_addr[stage], col_off + blk * B_LDS_ROW + half)
                    for blk in range_constexpr(2)
                    for half in (0, 512)
                ]
                lo = v[0].shuffle(v[1], list(range(8)))
                hi = v[2].shuffle(v[3], list(range(8)))
                return lo.shuffle(hi, list(range(16))), ds.mark(DS_B_OPS)

            def _stage_load_sa(stage, sm, k):
                off = (k * wmma_m_rep + sm * 2) * 16 * 4
                return lds_load_b32(stage_sa_addr[stage], off)[0], ds.mark(1)

            def _stage_load_sb(stage, sn, k):
                off = (sn * SC_WORDS + k * 32) * 4
                return lds_load_b32(stage_sb_addr[stage], off)[0], ds.mark(1)

            # Separate seed banks prevent WMMA source/address register coalescing.
            def _pipe(count, width):
                return [
                    [
                        [fx.make_rmem_tensor(width, fx.Int32) for _ in range_constexpr(K_STEPS)]
                        for _ in range_constexpr(count)
                    ]
                    for _ in range_constexpr(2)
                ]

            pipe_a = _pipe(half_m, 8)
            pipe_b = _pipe(half_n, 16)
            pipe_sa = _pipe(half_m // 2, 1)
            pipe_sb = _pipe(half_n, 1)
            seed_mark = [0, 0]  # DS mark after which each bank's seeds are complete

            def _scales_a(stage, sm0):
                return [
                    _stage_load_sa(stage, sm0 + sm, k)[0]
                    for sm in range_constexpr(half_m // 2)
                    for k in range_constexpr(K_STEPS)
                ]

            def _scales_b(stage, sn0):
                return [
                    _stage_load_sb(stage, sn0 + sn, k)[0]
                    for sn in range_constexpr(half_n)
                    for k in range_constexpr(K_STEPS)
                ]

            def _seed_a_frag(stage, bank, wm):
                for k in range_constexpr(K_STEPS):
                    pipe_a[bank][wm][k].store(_stage_load_a(stage, wm, k)[0])

            def _seed_b_frag(stage, bank, wn):
                for k in range_constexpr(K_STEPS):
                    pipe_b[bank][wn][k].store(_stage_load_b(stage, wn, k)[0])

            def _load_seed_a(stage, bank):
                for sm in range_constexpr(half_m // 2):
                    for k in range_constexpr(K_STEPS):
                        val, _ = _stage_load_sa(stage, sm, k)
                        pipe_sa[bank][sm][k].store(Vec.from_elements([val], fx.Int32))
                for wm in range_constexpr(half_m):
                    _seed_a_frag(stage, bank, wm)

            def _load_seed_b(stage, bank):
                for sn in range_constexpr(half_n):
                    for k in range_constexpr(K_STEPS):
                        val, _ = _stage_load_sb(stage, sn, k)
                        pipe_sb[bank][sn][k].store(Vec.from_elements([val], fx.Int32))
                for wn in range_constexpr(half_n):
                    _seed_b_frag(stage, bank, wn)

            def _pipe_scale(regs):
                return [reg.load()[0] for row in regs for reg in row]

            def _noop():
                pass

            def _sched_hint(groups):
                """One WMMA per DS burst across the produced group."""
                rocdl.sched_mfma(1)
                rocdl.sched_dsrd(2)
                for _ in range_constexpr(groups):
                    rocdl.sched_mfma(1)
                    rocdl.sched_dsrd(4)
                rocdl.sched_barrier(0)

            def _quadrant(
                wm0, wn0, act, wt, sa_k, sb_k, n_fast, need, produce, mid=None, mid_pos=0
            ):
                rocdl.sched_barrier(0)
                ds.wait(need)
                rocdl.sched_barrier(0)
                n_lead = len(produce)
                assert n_lead <= WMMA_PER_Q, f"{n_lead} producers exceed {WMMA_PER_Q} slots"
                for pos in range_constexpr(WMMA_PER_Q):
                    _mma_block_range(wm0, wn0, act, wt, sa_k, sb_k, pos, 1, n_fast)
                    if const_expr(mid is not None and pos == mid_pos):
                        mid()
                    if const_expr(pos < n_lead):
                        produce[pos]()
                    if const_expr(pos == n_lead - 1):
                        _sched_hint(n_lead - 1)
                rocdl.sched_barrier(0)
                return ds.issued

            def _compute_stage(
                stage,
                next_stage,
                bank,
                next_bank,
                future_slot,
                future_kt,
                fence_outstanding,
                has_next,
                parity,
            ):
                """One K-tile: four quadrants, each hosting one DS producer group."""
                a_top, b_left = pipe_a[bank], pipe_b[bank]
                sa_top, sb_left = _pipe_scale(pipe_sa[bank]), _pipe_scale(pipe_sb[bank])

                a_bottom = [[None] * K_STEPS for _ in range_constexpr(half_m)]
                b_right = [[None] * K_STEPS for _ in range_constexpr(half_n)]
                sa_bottom, sb_right = [], []

                def _produce_b_right():
                    out = [lambda: sb_right.extend(_scales_b(stage, half_n))]
                    for wn in range_constexpr(half_n):
                        for k in range_constexpr(K_STEPS):

                            def _go(wn=wn, k=k):
                                b_right[wn][k] = _rmem(
                                    16, _stage_load_b(stage, half_n + wn, k)[0]
                                )

                            out.append(_go)
                    return out

                def _produce_a_bottom():
                    out = [lambda: sa_bottom.extend(_scales_a(stage, half_m // 2))]
                    for wm in range_constexpr(half_m):
                        for k in range_constexpr(K_STEPS):

                            def _go(wm=wm, k=k):
                                a_bottom[wm][k] = _rmem(
                                    8, _stage_load_a(stage, half_m + wm, k)[0]
                                )

                            out.append(_go)
                    return out

                def _produce_seed_a():
                    def _scales():
                        if const_expr(has_next):
                            for sm in range_constexpr(half_m // 2):
                                for k in range_constexpr(K_STEPS):
                                    val, _ = _stage_load_sa(next_stage, sm, k)
                                    pipe_sa[next_bank][sm][k].store(
                                        Vec.from_elements([val], fx.Int32)
                                    )

                    out = [_scales]
                    for wm in range_constexpr(half_m):

                        def _go(wm=wm):
                            if const_expr(has_next):
                                _seed_a_frag(next_stage, next_bank, wm)

                        out.append(_go)
                    return out

                def _produce_seed_b():
                    def _scales():
                        if const_expr(has_next):
                            for sn in range_constexpr(half_n):
                                for k in range_constexpr(K_STEPS):
                                    val, _ = _stage_load_sb(next_stage, sn, k)
                                    pipe_sb[next_bank][sn][k].store(
                                        Vec.from_elements([val], fx.Int32)
                                    )

                    out = [_scales]
                    for wn in range_constexpr(half_n):

                        def _go(wn=wn):
                            if const_expr(has_next):
                                _seed_b_frag(next_stage, next_bank, wn)

                        out.append(_go)
                    return out

                TL, TR = (0, 0), (0, half_n)
                BL, BR = (half_m, 0), (half_m, half_n)
                if const_expr(parity == 0):
                    plan = [
                        (TL, a_top, b_left, _produce_b_right),
                        (TR, a_top, b_right, _produce_a_bottom),
                        (BL, a_bottom, b_left, _produce_seed_a),
                        (BR, a_bottom, b_right, _produce_seed_b),
                    ]
                else:
                    plan = [
                        (TL, a_top, b_left, _produce_a_bottom),
                        (BL, a_bottom, b_left, _produce_b_right),
                        (TR, a_top, b_right, _produce_seed_b),
                        (BR, a_bottom, b_right, _produce_seed_a),
                    ]

                need, mid = seed_mark[bank], None
                for idx in range_constexpr(4):
                    (wm0, wn0), act, wt, produce = plan[idx]
                    if const_expr(idx == 2):
                        need = ds.issued
                        refill = const_expr(future_kt is not None)
                        prepared = (
                            _prepare_tdm(future_slot, future_kt)
                            if const_expr(refill)
                            else None
                        )
                        rocdl.sched_barrier(0)
                        ds.wait(need)
                        rocdl.sched_barrier(0)
                        if const_expr(has_next):
                            pipeline_fence_signal(
                                outstanding=fence_outstanding, use_cluster=False
                            )

                            def mid():
                                rocdl.sched_barrier(0)
                                pipeline_fence_wait(use_cluster=False)
                                if const_expr(refill):
                                    tdm_ops.tensor_load_2d(prepared)
                                rocdl.sched_barrier(0)

                    _quadrant(
                        wm0,
                        wn0,
                        act,
                        wt,
                        sa_top + sa_bottom,
                        sb_left + sb_right,
                        idx > 0,
                        need,
                        produce()
                        if const_expr(idx != 2)
                        else [_noop] * (FENCE_WAIT_POS + 1) + produce(),
                        mid,
                        FENCE_WAIT_POS,
                    )
                    if const_expr(idx == 2):
                        mid = None
                    need = ds.issued if const_expr(idx < 2) else need
                seed_mark[next_bank] = ds.issued

            # Keep all slots in flight and statically expand one revolution.
            for i in range_constexpr(num_buffers):
                tdm_ops.tensor_load_2d(_prepare_tdm(i, i))
            pipeline_fence(outstanding=num_buffers - 1, use_cluster=False)
            _load_seed_a(0, 0)
            _load_seed_b(0, 0)
            seed_mark[0] = ds.issued

            n_steady = K_TILES - num_buffers

            def _run_all(parity):
                for rev in range(n_steady // num_buffers):
                    do_sync = (rev % cluster_sync_revs) == (cluster_sync_revs - 1)
                    for s in range_constexpr(num_buffers):
                        kt = rev * num_buffers + s
                        _compute_stage(
                            s,
                            (s + 1) % num_buffers,
                            s % 2,
                            (s + 1) % 2,
                            s,
                            kt + num_buffers,
                            num_buffers - 2,
                            True,
                            parity,
                        )
                    if do_sync:
                        cluster.cluster_barrier()
                for j in range_constexpr(num_buffers):
                    # No refills left, so the drain ratchets the TDM allowance
                    # down: stage j seeds slot j+1, whose prologue load must
                    # have landed by then.
                    _compute_stage(
                        j,
                        (j + 1) % num_buffers,
                        j % 2,
                        (j + 1) % 2,
                        j,
                        None,
                        num_buffers - 2 - j,
                        j < num_buffers - 1,
                        parity,
                    )

            wave_parity = fx.Int32(rocdl.readfirstlane(T.i32, wave % 2))
            if wave_parity == 0:
                _run_all(0)
            else:
                _run_all(1)

            rocdl.s_wait_dscnt(0)
            # acc (wm, wn) is 32 N-cols x 16 M-rows: lane holds M = wm*16+lane16
            # and, per 8-wide half, N = wn*32 + half*16 + kgrp*8 + v. Splitting
            # each into halves gives the 16-col fragment order the epilogue uses.
            accs = []
            for idx in range_constexpr(n_acc):
                acc = Vec(c_frags[idx].load())
                for fragment in range_constexpr(2):
                    accs.append(
                        Vec.from_elements(
                            [acc[fragment * 8 + i] for i in range_constexpr(8)],
                            fx.Float32,
                        ).ir_value()
                    )

            # The epilogue restages C in this arena, so drain the whole cluster:
            # peer multicast loads are pairwise matched with ours.
            pipeline_fence(outstanding=0, use_cluster=True)

            stC_idx = fx.Index(fx.ptrtoint(base_ptr))
            neg_limit = fx.Float32(0.0) - f32_swiglu_limit
            # Clamp bounds are wave-uniform, so the backend parks them in SGPRs
            # and every v_med3 then reads TWO different scalars through the
            # narrow scalar operand port. Forcing them into VGPRs makes all 512
            # med3 all-VGPR and is worth 28 us (604.7 -> 576.3).
            if const_expr(_EPI_CLAMP_VGPR):
                gate_lo = make_vgpr_opaque(fx.Float32(-3.4028234663852886e38))
                clamp_hi = make_vgpr_opaque(f32_swiglu_limit)
                clamp_lo = make_vgpr_opaque(neg_limit)
            else:
                gate_lo = None
                clamp_hi, clamp_lo = f32_swiglu_limit, neg_limit
            is_swiglu = stage1_act == 2
            is_situv2 = stage1_act == 3
            situ_c = (
                situv2_consts(f32_situ_beta, f32_situ_linear_beta)
                if const_expr(is_situv2)
                else None
            )

            if const_expr(is_fp4_quant):
                # Fused activation + MX quant: payload to LDS, e8m0 to global.
                i8_ptr_g = fx.PointerType.get(
                    elem_ty=fx.Int8.ir_type,
                    address_space=fx.AddressSpace.Global,
                    alignment=1,
                )
                scale_ptr = fx.recast_iter(i8_ptr_g, fx.get_iter(arg_quant_scale))
                is_kgrp0 = fx.Int32(kgrp) == fx.Int32(0)
                # i32_n is the pre-activation gate+up width; the quantized output
                # has half the columns and one scale dword per K128.
                q_dst_scale_dwpr = i32_n // 256
                QUANT_ROWS_PER_TILE = quant_wmma_rep * 16
                QRPT_LOG2 = int(math.log2(QUANT_ROWS_PER_TILE))
                N_MX_BLKS = output_n_rep // WN_PER_MX_BLOCK
                # Rows are batched _EPI_ACT_WM 16-row blocks at a time. The
                # activation is elementwise, so batch width is purely a
                # scheduling knob -- but at 1008 VGPRs this kernel runs one wave
                # per SIMD, so nothing else is resident to hide TRANS latency
                # and the epilogue is latency-bound: measured 653 us at 32
                # values per batch vs 958 us at 16, with byte-identical VGPR
                # count and instruction count. Widen until it stops paying.
                WM_G = _EPI_ACT_WM
                for wm0 in range_constexpr(wmma_m_rep // WM_G):
                    # A 16-row block entirely past this expert's rows is
                    # OOB-clamped away at the store, so skip its work. Wave
                    # uniform: one scalar branch, not per-lane masking. Rows
                    # increase with wm, so testing the group's first row skips
                    # only groups that are wholly out of range.
                    if wmb + wm0 * WM_G * 16 < mn_oob:
                        rows = []
                        for wi in range_constexpr(WM_G):
                            wm = wm0 * WM_G + wi
                            row_rel = wmb + wm * 16 + lane16
                            row_i32 = fx.Int32(blk_m + row_rel)
                            rows.append(
                                (
                                    wm,
                                    row_rel,
                                    row_i32 >> QRPT_LOG2,
                                    (row_i32 & (QUANT_ROWS_PER_TILE - 1)) >> 4,
                                    row_i32 & (QUANT_ROWS_PER_TILE - 1) & 15,
                                )
                            )

                        e8m0_bytes = {}
                        mx_blk_is = {}
                        for grp in range_constexpr(N_MX_BLKS // _EPI_ACT_BLKS):
                            pairs = []
                            for wi in range_constexpr(WM_G):
                                wm = wm0 * WM_G + wi
                                for b in range_constexpr(_EPI_ACT_BLKS):
                                    mx_blk = grp * _EPI_ACT_BLKS + b
                                    for sub_wn in range_constexpr(WN_PER_MX_BLOCK):
                                        wn = mx_blk * WN_PER_MX_BLOCK + sub_wn
                                        acc = Vec(accs[wm * output_n_rep + wn])
                                        for p in range_constexpr(4):
                                            pairs.append((acc[2 * p], acc[2 * p + 1]))

                            _clamp_on = bool(act_has_limit) and not (_EPI_PROBE & 1)
                            if const_expr(_EPI_PROBE & 2):
                                # PROBE: keep the clamp and the final *u, drop
                                # only mul-0.5 / v_tanh / v_fma. Same value
                                # count, so the pack path is untouched.
                                if const_expr(_clamp_on):
                                    batch_vals = [
                                        fclamp_f32(
                                            pairs[i][0], neg_limit, f32_swiglu_limit
                                        )
                                        * fclamp_f32(
                                            pairs[i][1], neg_limit, f32_swiglu_limit
                                        )
                                        for i in range_constexpr(len(pairs))
                                    ]
                                else:
                                    batch_vals = [
                                        pairs[i][0] * pairs[i][1]
                                        for i in range_constexpr(len(pairs))
                                    ]
                            elif const_expr(is_situv2):
                                batch_vals = batched_situv2(
                                    pairs,
                                    consts=situ_c,
                                    range_constexpr=range_constexpr,
                                )
                            else:
                                batch_vals = batched_silu_swiglu(
                                    pairs,
                                    swiglu=is_swiglu,
                                    limit_f32=clamp_hi,
                                    neg_limit_f32=clamp_lo,
                                    range_constexpr=range_constexpr,
                                    barriers=_EPI_ACT_BARRIERS,
                                    has_limit=_clamp_on,
                                    use_hw_tanh=_EPI_HW_TANH,
                                    late_u_clamp=_EPI_LATE_U_CLAMP,
                                    gate_lo_f32=gate_lo,
                                )

                            vals_per_blk = WN_PER_MX_BLOCK * 4
                            for wi in range_constexpr(WM_G):
                                wm = wm0 * WM_G + wi
                                row_rel = rows[wi][1]
                                for b in range_constexpr(_EPI_ACT_BLKS):
                                    mx_blk = grp * _EPI_ACT_BLKS + b
                                    base = (wi * _EPI_ACT_BLKS + b) * vals_per_blk
                                    all_vals = batch_vals[base : base + vals_per_blk]
                                    if const_expr(_EPI_PROBE & 4):
                                        # PROBE: constant block scale -- drops
                                        # the |.| max tree, its peer shuffle and
                                        # the e8m0 assembly, keeps both uses.
                                        scale_f32 = fx.Float32(1.0)
                                        e8m0_byte = arith.trunci(
                                            T.i8, fx.Int32(127).ir_value()
                                        )
                                    else:
                                        scale_f32, e8m0_byte = (
                                            emit_amax_e8m0_native_scale(
                                                all_vals,
                                                wave_size=WAVE,
                                                dtype=MxDtype.FP4_E2M1,
                                            )
                                        )
                                    mx_col = (
                                        blk_n + wnb + mx_blk * WN_PER_MX_BLOCK * 16
                                    )
                                    e8m0_bytes[(wi, mx_blk)] = e8m0_byte
                                    mx_blk_is[(wi, mx_blk)] = fx.Int32(mx_col) >> 6

                                    for sub_wn in range_constexpr(WN_PER_MX_BLOCK):
                                        wn = mx_blk * WN_PER_MX_BLOCK + sub_wn
                                        local_vals = all_vals[
                                            sub_wn * 4 : sub_wn * 4 + 4
                                        ]
                                        if const_expr(_EPI_PACK_B16_STORE):
                                            # Each half-wave stores only its own
                                            # 4 fp4 values (2 bytes), so no peer
                                            # data is needed at all: the pk8 is
                                            # fed our 2 dwords twice and only its
                                            # low 16 bits are kept. Trades 2
                                            # v_permlanex16 per pk8 -- the most
                                            # expensive op in the epilogue -- for
                                            # a narrower store from all 32 lanes
                                            # instead of a b32 from half of them.
                                            own_dw = (
                                                Vec.from_elements(
                                                    local_vals, fx.Float32
                                                )
                                                .to(fx.BFloat16)
                                                .bitcast(fx.Int32)
                                            )
                                            src = Vec.from_elements(
                                                [
                                                    own_dw[0],
                                                    own_dw[1],
                                                    own_dw[0],
                                                    own_dw[1],
                                                ],
                                                fx.Int32,
                                            ).bitcast(fx.BFloat16)
                                        elif const_expr(_EPI_PACK_BF16_XCHG):
                                            # Convert to bf16 BEFORE the peer
                                            # exchange, not after. The pk8 pack
                                            # wants 8 bf16 = 4 dwords, of which
                                            # 4 lanes' worth are ours and 4 the
                                            # kgrp peer's. Exchanging the two
                                            # already-packed dwords costs 2
                                            # v_permlanex16 instead of 4 on
                                            # loose f32, and each lane converts
                                            # only its own 4 values instead of
                                            # all 8: half the cross-lane traffic
                                            # and half the v_cvt_pk_bf16_f32.
                                            own_dw = (
                                                Vec.from_elements(
                                                    local_vals, fx.Float32
                                                )
                                                .to(fx.BFloat16)
                                                .bitcast(fx.Int32)
                                            )
                                            if const_expr(_EPI_PROBE & 8):
                                                peer_dw = own_dw  # PROBE
                                            else:
                                                peer_dw = own_dw.shuffle_xor(16, WAVE)
                                            src = Vec.from_elements(
                                                [
                                                    own_dw[0],
                                                    own_dw[1],
                                                    peer_dw[0],
                                                    peer_dw[1],
                                                ],
                                                fx.Int32,
                                            ).bitcast(fx.BFloat16)
                                        else:
                                            if const_expr(_EPI_PROBE & 8):
                                                peer_vals = list(local_vals)  # PROBE
                                            else:
                                                peer_vals = [
                                                    fx.Float32(value).shuffle_xor(
                                                        16, WAVE
                                                    )
                                                    for value in local_vals
                                                ]
                                            src = Vec.from_elements(
                                                local_vals + peer_vals, fx.Float32
                                            ).to(fx.BFloat16)
                                        packed_i32 = emit_cvt_scalef32_pk8_fp4_bf16(
                                            src.ir_value(),
                                            scale_f32,
                                            i32_ty=T.i32,
                                        )
                                        col_fp4 = (wnb + wn * 16) // 4
                                        if const_expr(_EPI_PACK_B16_STORE):
                                            lds_store_b16_raw(
                                                stC_idx,
                                                fx.Int32(row_rel * STORE_N + col_fp4)
                                                + fx.Int32(kgrp) * 2,
                                                packed_i32,
                                            )
                                        elif kgrp == 0:
                                            lds_store_b32(
                                                stC_idx,
                                                row_rel * STORE_N + col_fp4,
                                                Vec.from_elements(
                                                    [packed_i32], fx.Int32
                                                ),
                                            )

                        # Preshuffled e8m0 scale: one branch per 16-row block.
                        for wi in range_constexpr(WM_G):
                            _, row_rel, scale_tile, wmma_row, scale_lane = rows[wi]
                            if row_rel < mn_oob and is_kgrp0:
                                for mx_blk in range_constexpr(N_MX_BLKS):
                                    blk_i = mx_blk_is[(wi, mx_blk)]
                                    scale_dw = blk_i >> 2
                                    byte_in_dw = blk_i & 3
                                    dst_byte = (
                                        (
                                            (scale_tile * q_dst_scale_dwpr + scale_dw)
                                            * quant_wmma_rep
                                            + wmma_row
                                        )
                                        * 16
                                        + scale_lane
                                    ) * 4 + byte_in_dw
                                    fx.ptr_store(
                                        e8m0_bytes[(wi, mx_blk)], scale_ptr + dst_byte
                                    )
            else:
                if const_expr(has_bias):
                    bias_ptr_type = fx.PointerType.get(
                        elem_ty=out_elem,
                        address_space=fx.AddressSpace.Global,
                        alignment=2,
                    )
                    bias_map = fx.recast_iter(bias_ptr_type, arg_bias)
                for wm in range_constexpr(wmma_m_rep):
                    row_rel = wmb + wm * 16 + lane16
                    for wn in range_constexpr(output_n_rep):
                        col_rel = wnb + wn * 16 + kgrp * 8
                        acc = Vec(accs[wm * output_n_rep + wn])
                        if const_expr(has_bias):
                            acc = acc + Vec(
                                fx.ptr_load(
                                    bias_map + expert * i32_n + col_rel,
                                    result_type=T.vec(8, out_elem),
                                )
                            ).to(fx.Float32)
                        if const_expr(stage1_act):
                            if const_expr(is_situv2):
                                act_vals = [
                                    fused_situv2_elem(
                                        acc[2 * p], acc[2 * p + 1], consts=situ_c
                                    )
                                    for p in range_constexpr(4)
                                ]
                            else:
                                act_vals = [
                                    fused_silu_swiglu_elem(
                                        acc[2 * p],
                                        acc[2 * p + 1],
                                        swiglu=is_swiglu,
                                        limit_f32=f32_swiglu_limit,
                                        neg_limit_f32=neg_limit,
                                        has_limit=bool(act_has_limit),
                                    )
                                    for p in range_constexpr(4)
                                ]
                            hv = Vec.from_elements(act_vals, fx.Float32).to(oc)
                            lds_store_b64(
                                stC_idx,
                                (row_rel * STORE_N + col_rel // 2) * 2,
                                hv.bitcast(fx.Int32).ir_value(),
                            )
                        else:
                            hv = Vec.from_elements(
                                [acc[i] for i in range_constexpr(8)], fx.Float32
                            ).to(oc)
                            lds_store_b128(
                                stC_idx,
                                (row_rel * STORE_PITCH + col_rel) * 2,
                                hv.bitcast(fx.Int32).ir_value(),
                            )

            # -- staged LDS tile -> global --
            if const_expr(is_fp4_quant):
                # The e8m0 stores are still in flight but the tile store reads
                # LDS only, so a dscnt-only barrier lets them retire past it.
                rocdl.s_wait_dscnt(0)
                rocdl.s_barrier_signal(-1)
                rocdl.s_barrier_wait(-1)
            else:
                workgroup_barrier(use_cluster=False)

            if const_expr(stage1_act):
                out_divisor = 4 if is_fp4_quant else 2
                out_stride = i32_n // out_divisor
                out_col_off = blk_n64 // out_divisor
            else:
                out_stride = i32_n
                out_col_off = blk_n64
            oc_store = fx.Int8 if const_expr(is_fp4_quant) else oc
            c_iter = (
                fx.recast_iter(fx.Int8, fx.get_iter(arg_c))
                if const_expr(is_fp4_quant)
                else fx.get_iter(arg_c)
            )
            c_off_rt = blk_m64 * fx.Int64(out_stride) + out_col_off
            if const_expr(STORE_PAD == 0):
                gtC = _gv(c_iter, c_off_rt, (tile_m, STORE_N), (STORE_N, 1))
                atomC = fx.rocdl.make_tdm_atom(
                    gtC, [mn_oob, None], strides=[out_stride, None], num_warps=num_waves
                )
                src = _view(
                    fx.recast_iter(oc_store, base_ptr), (tile_m, STORE_N), (STORE_N, 1)
                )
            else:
                # The LDS tile is (tile_m, STORE_PITCH) dense; the inner OOB
                # extent clamps to STORE_N so the pad never reaches global.
                gtC = _gv(c_iter, c_off_rt, (tile_m, STORE_PITCH), (out_stride, 1))
                atomC = fx.rocdl.make_tdm_atom(
                    gtC,
                    [mn_oob, STORE_N],
                    strides=[out_stride, None],
                    num_warps=num_waves,
                )
                src = _view(
                    fx.recast_iter(oc_store, base_ptr),
                    (tile_m, STORE_PITCH),
                    (STORE_PITCH, 1),
                )
            fx.copy(atomC, src, gtC)
            if const_expr(is_fp4_quant):
                rocdl.s_wait_storecnt(0)
            tdm_ops.tensor_wait(0)

    gx = (i32_m + (tile_m - 1)) // tile_m
    gy = (N + (tile_n - 1)) // tile_n
    gx = (((gx > 0).select(gx, fx.Int32(1)) + (cluster_m - 1)) // cluster_m) * cluster_m
    # Split gx exactly, so no workgroup is left over to recompute a duplicate
    # tile -- and split it in CLUSTERS, not workgroups. grid.x must be a whole
    # number of clusters: a trailing partial cluster has multicast peers that
    # were never launched, and its members wait on them forever. gx is already a
    # multiple of cluster_m so the cluster count is exact, and capping the run at
    # a POWER OF TWO keeps it a divisor of cx (= pow2 * odd), which is what keeps
    # m_chunks exact. For the tuned 4x4 this reproduces the old split exactly.
    cx = gx // cluster_m
    run_max = 1 << (max(1, m_run_max // cluster_m).bit_length() - 1)
    run_min = max(1, m_run_min // cluster_m)
    pow2 = cx & -cx
    capped = (pow2 < run_max).select(pow2, fx.Int32(run_max))
    c_run = ((cx > run_max) & (pow2 >= run_min)).select(capped, cx)
    m_run = c_run * cluster_m
    m_chunks = cx // c_run
    # gy % cluster_n and the per-expert tile_m*cluster_m alignment are the
    # caller's to enforce; inside @flyc.jit N is traced, so a Python check here
    # would become a device branch rather than a host-side assert.
    kernel(
        arg_c,
        arg_a,
        arg_b,
        arg_scale_a,
        arg_scale_b,
        arg_m_tile_map,
        arg_bias,
        arg_quant_scale,
        i32_m,
        N,
        # Constant-valued but passed as a kernel operand, not folded in: a
        # Python int here would make K_TILES compile-time and unroll the
        # revolution loop into thousands of WMMAs.
        fx.Int32(K),
        f32_swiglu_limit,
        f32_situ_beta,
        f32_situ_linear_beta,
        value_attrs={"rocdl.cluster_dims": f"{cluster_m},{cluster_n},1"},
    ).launch(
        grid=(m_run, gy, m_chunks),
        block=(block, 1, 1),
        stream=stream,
        cluster=(cluster_m, cluster_n, 1),
    )


launch_gemm_a4w4_moe.compile_hints["llvm_options"] = {
    "amdgpu-expert-scheduling-mode": AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE,
    "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
    "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
}
