"""hstu_attention_fwd - FlyDSL kernel

out_i = (1/N) * sum_j valid(i,j) * silu(alpha * q_i * k_j^T) * v_j

HSTU forward for windowed symmetric heads and selected full-causal heads. GEMM2 consumes P as
operand A and row-major V as operand B, so V stages without a transposed LDS scatter. K stages by
dword DMA; V loads to registers, waits with counted vmcnt, and then publishes to LDS for GEMM2.
Row coordinates are cast to i64 so view address arithmetic cannot overflow on large packed tensors.

Inputs/outputs:
  - q,k (L, H, attn_dim); v,out (L, H, hidden_dim): packed jagged, `dtype`. Passed in their
    native rank-3 layout (q/k/v must be contiguous); no host-side flatten to (L, H*dim).
  - seq_offsets (Z+1) i32, num_targets (Z) i32.

Paths:
  - windowed (max_attn_len > 0): sliding-window lower bound skips fully-masked low KV tiles.
  - full-causal (max_attn_len == 0): causal upper bound only.
  - contextual (contextual_seq_len > 0): id shift+clamp, prefix-opener term, and the prefix
    query block opens its KV range (upper→seq_len, lower→0) to see the whole prefix.

Constraints:
  - causal; contextual prefix requires causal.
  - dtype in {f16, bf16}.
  - attn_dim % 16 == 0, hidden_dim % 16 == 0; (batch*num_heads) % 8 == 0; tile divisibility is a
    build-time contract.
  - fast/unsafe FP math: not strict IEEE-754.
"""

import functools
import math as host_math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl
from flydsl.expr.arith import ArithValue
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SMEM_CAPACITY_MAP, SmemAllocator, SmemPtr


def _dtype_to_elem_type(dtype_str: str):
    if dtype_str == "f16":
        return fx.Float16
    if dtype_str == "bf16":
        return fx.BFloat16
    raise ValueError(f"unsupported dtype: {dtype_str!r} (expected 'f16' or 'bf16')")


# ---- Kernel Geometry Constants ----

WARP_SIZE = 64
NUM_GRID_GROUPS = 8  # grid decoded group-major for locality
MFMA_M = 16
MFMA_N = 16
MFMA_K = 16
MFMA_LANE_K = 4
MFMA_ELEMS_PER_LANE = (MFMA_M * MFMA_N) // WARP_SIZE  # 4 f32 per lane

# K LDS swizzle and DMA granule are arch-conditional: bank conflict period
# differs between CDNA3 (128 B) and CDNA4 (256 B). Mask must stay < 64 elements so col ^ mask never
# leaves the row for any supported HEAD_DIM_K (including non-power-of-2 192).


def _arch_dma_params(arch: str | None = None):
    """Arch-conditional (DMA_BYTES, DMA_ELEMS, K_SWZ_ROWS, K_SWZ_SHIFT).

    gfx942 (CDNA3): dword DMA (4 B/lane), 128-byte bank period -> (16, 2) covers a 64-element block.
    gfx950 (CDNA4): dwordx4 DMA (16 B/lane), 256-byte bank period -> (8, 3) covers a 64-element block.
    """
    if arch is None:
        arch = get_rocm_arch()
    if (arch or "").startswith("gfx942"):
        dma_bytes = 4  # CDNA3 dword
        k_swz_rows, k_swz_shift = 16, 2
    else:
        dma_bytes = 16  # CDNA4 dwordx4
        k_swz_rows, k_swz_shift = 8, 3
    return dma_bytes, dma_bytes // 2, k_swz_rows, k_swz_shift


# V LDS is fully contiguous row-major [BLOCK_N, hidden_dim]: the dword buffer_load_lds scatters
# lanes contiguously from a wave-uniform base, so the LDS layout must match the global row-major
# fetch order.


@functools.lru_cache(maxsize=16384)
def lds_cap_bytes(arch: str | None = None) -> int:
    default_lds_cap_bytes = 65536

    if arch is None:
        arch = get_rocm_arch()
    return SMEM_CAPACITY_MAP.get(arch, default_lds_cap_bytes)


_LOG2E = host_math.log2(host_math.e)


def _waitcnt_vm_n(n: int):
    """Emit s_waitcnt vmcnt(n) only (lgkmcnt=63, expcnt=7)."""

    # s_waitcnt vmcnt is split lo[3:0] @ bit 0, hi[5:4] @ bit 14.
    # lgkmcnt(63) + expcnt(7) stay maximal so only vmcnt is constrained; V register loads remain
    # outstanding while K DMA is drained.
    vmcnt_lo_mask = 0xF
    lgkmcnt_expcnt_base = 0x3F70
    vmcnt_hi_shift = 14
    vmcnt_hi_mask = 0x3
    val = (n & vmcnt_lo_mask) | lgkmcnt_expcnt_base | (((n >> 4) & vmcnt_hi_mask) << vmcnt_hi_shift)
    rocdl.s_waitcnt(val)


def validate_hstu_attention_fwd(
    num_heads: int,
    head_dim: int,
    hidden_dim: int,
    batch: int,
    causal: bool,
    max_attn_len: int,
    contextual_seq_len: int,
    has_targets: bool,
    alpha: float,
    dtype_str: str,
    max_seq_len: int,
    *,
    block_m: int,
    block_n: int,
    num_waves: int,
    waves_per_eu: int,
    arch: str | None = None,
) -> None:
    if arch is None:
        arch = get_rocm_arch()
    if not arch.startswith("gfx942") and not arch.startswith("gfx950"):
        raise ValueError(f"hstu attention fwdunsupported arch: {arch!r} (expected 'gfx942' or 'gfx950')")

    if dtype_str not in {"f16", "bf16"}:
        raise ValueError(f"unsupported dtype: {dtype_str!r} (expected 'f16' or 'bf16')")
    if not causal:
        raise ValueError("hstu_attention_fwd only supports causal attention")
    if contextual_seq_len < 0:
        raise ValueError(f"contextual_seq_len must be non-negative, got {contextual_seq_len}")
    if max_attn_len < 0:
        raise ValueError(f"max_attn_len must be non-negative, got {max_attn_len}")
    if batch <= 0:
        raise ValueError(f"batch must be positive, got {batch}")
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}")
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
    if not host_math.isfinite(alpha):
        raise ValueError(f"alpha must be finite, got {alpha}")
    if head_dim <= 0 or head_dim % MFMA_K != 0:
        raise ValueError(f"head_dim must be positive and a multiple of MFMA_K={MFMA_K}, got {head_dim}")
    if hidden_dim <= 0 or hidden_dim % MFMA_M != 0:
        raise ValueError(f"hidden_dim must be positive and a multiple of MFMA_M={MFMA_M}, got {hidden_dim}")
    if (batch * num_heads) % NUM_GRID_GROUPS != 0:
        raise ValueError(f"require (batch*num_heads) % {NUM_GRID_GROUPS} == 0, got {batch * num_heads}")

    if block_m <= 0:
        raise ValueError(f"block_m must be positive, got {block_m}")
    if block_n <= 0:
        raise ValueError(f"block_n must be positive, got {block_n}")
    if num_waves <= 0:
        raise ValueError(f"num_waves must be positive, got {num_waves}")
    if waves_per_eu < 0:
        raise ValueError(f"waves_per_eu must be non-negative, got {waves_per_eu}")
    if block_m % (num_waves * MFMA_M) != 0:
        raise ValueError(f"block_m {block_m} must be a multiple of num_waves*MFMA_M ({num_waves * MFMA_M})")
    if block_n % MFMA_M != 0:
        raise ValueError(f"block_n {block_n} must be a multiple of MFMA_M={MFMA_M}")

    _, dma_elems, _, _ = _arch_dma_params()
    block_threads = num_waves * WARP_SIZE
    elems_per_dma_pass = block_threads * dma_elems
    head_dim_k = ((head_dim + 63) // 64) * 64
    if (block_n * head_dim_k) % elems_per_dma_pass != 0:
        raise ValueError("K DMA tile does not divide the dword DMA pass evenly")
    if (block_n * hidden_dim) % elems_per_dma_pass != 0:
        raise ValueError("V DMA tile does not divide the dword DMA pass evenly")

    vec_v = 8 if (hidden_dim % 8 == 0 and (block_n * hidden_dim) % (block_threads * 8) == 0) else dma_elems
    threads_per_row_v = hidden_dim // vec_v
    if block_threads % threads_per_row_v != 0:
        raise ValueError(f"block_threads={block_threads} must be divisible by threads_per_row_v={threads_per_row_v}")
    rows_per_batch_v = block_threads // threads_per_row_v
    if not (block_n % rows_per_batch_v == 0 or rows_per_batch_v > block_n):
        raise ValueError(f"rows_per_batch_v={rows_per_batch_v} must divide block_n={block_n}, unless rows_per_batch_v > block_n")

    lds_cap = lds_cap_bytes()
    lds_bytes = block_n * head_dim_k * 2 + block_n * hidden_dim * 2
    if lds_bytes > lds_cap:
        raise ValueError(f"LDS tile {lds_bytes} B exceeds the {lds_cap} B budget")


@functools.lru_cache(maxsize=16384)
def build_hstu_attention_fwd(
    num_heads: int,
    head_dim: int,
    hidden_dim: int,
    batch: int,
    causal: bool,
    max_attn_len: int,
    contextual_seq_len: int,
    has_targets: bool,
    alpha: float,
    dtype_str: str,
    max_seq_len: int,
    *,  # keyword-only tunables
    block_m: int = 64,
    block_n: int = 16,
    num_waves: int = 2,
    waves_per_eu: int = 0,
):
    validate_hstu_attention_fwd(
        num_heads,
        head_dim,
        hidden_dim,
        batch,
        causal,
        max_attn_len,
        contextual_seq_len,
        has_targets,
        alpha,
        dtype_str,
        max_seq_len,
        block_m=block_m,
        block_n=block_n,
        num_waves=num_waves,
        waves_per_eu=waves_per_eu,
    )

    BLOCK_M = block_m
    BLOCK_N = block_n
    NUM_WAVES = num_waves
    BLOCK_THREADS = NUM_WAVES * WARP_SIZE
    ROWS_PER_WAVE = BLOCK_M // NUM_WAVES
    Q_SUBTILES = ROWS_PER_WAVE // MFMA_M  # query sub-tiles per wave
    KV_SUBTILES = BLOCK_N // MFMA_N  # KV sub-tiles
    WAVES_PER_EU = waves_per_eu

    assert num_waves > 0 and block_m % (num_waves * MFMA_M) == 0
    assert block_n % MFMA_M == 0
    assert head_dim % MFMA_K == 0
    assert hidden_dim % MFMA_M == 0
    assert (batch * num_heads) % NUM_GRID_GROUPS == 0

    # Arch-conditional DMA + K LDS swizzle period
    DMA_BYTES, DMA_ELEMS, K_SWZ_ROWS, K_SWZ_SHIFT = _arch_dma_params()

    elem_dtype = _dtype_to_elem_type(dtype_str)
    is_bf16 = dtype_str == "bf16"
    has_window = max_attn_len > 0
    has_contextual = contextual_seq_len > 0

    K_STEPS = head_dim // MFMA_K  # real 16-wide contraction steps (Q side)
    # The XOR swizzle formula (k_swz_col) has period 64 elements. When K_STRIDE < 64
    # (head_dim % 64 != 0) the swizzled column lands outside [0, K_STRIDE), producing wrong LDS
    # addresses on both the DMA write and the LDS read. GEMM1 is unaffected (Q is register-resident),
    # but GEMM2's A-operand (P packs read back through the swizzled K LDS) gets corrupted data.
    # Round head_dim up to a 64-aligned stride so the swizzle stays in-row by construction.
    # Extra K columns over-fetch out-of-range data (buffer bounds -> 0) and pair with a zero Q
    # operand, contributing nothing. The aligned case leaves HEAD_DIM_K == head_dim.
    HEAD_DIM_K = ((head_dim + 63) // 64) * 64
    K_STEPS_K = HEAD_DIM_K // MFMA_K  # padded steps (K side); always a multiple of 4
    D_CHUNKS = hidden_dim // MFMA_M  # O accumulator / GEMM2 chunks

    num_q_tiles = (max_seq_len + BLOCK_M - 1) // BLOCK_M
    hz_per_group = (batch * num_heads) // NUM_GRID_GROUPS

    stride_qk_n = num_heads * head_dim

    K_STRIDE = HEAD_DIM_K  # no PAD_K - XOR swizzle replaces it; 64-aligned so the swizzle stays in-row
    k_lds_bytes = BLOCK_N * K_STRIDE * 2

    # V LDS: row-major V[kv, d], no row pad. The contiguous lane scatter matches the
    # natural global layout.
    V_STRIDE = hidden_dim
    v_lds_bytes = BLOCK_N * V_STRIDE * 2

    # K DMA tiling (dword passes).
    k_tile_elems = BLOCK_N * K_STRIDE
    elems_per_dma_pass = BLOCK_THREADS * DMA_ELEMS
    assert k_tile_elems % elems_per_dma_pass == 0
    NUM_DMA_K = k_tile_elems // elems_per_dma_pass
    PAIRS_PER_ROW_K = K_STRIDE // DMA_ELEMS

    # V tile divisibility (the DMA pass also gates the register-load tiling below).
    v_tile_elems = BLOCK_N * hidden_dim
    assert v_tile_elems % elems_per_dma_pass == 0

    # V register-prefetch tiling. Each lane reads VEC_V contiguous d elements
    # of one V row, coalesced (dwordx{VEC_V//2}). One pass = ROWS_PER_BATCH_V rows; NUM_BATCHES_V
    # passes cover the BLOCK_N×hidden_dim tile. Coalesced AND overlappable (register dest, not
    # buffer_load_lds), so the load can stay in flight across GEMM1 under a counted vmcnt.
    VEC_V = 8 if (hidden_dim % 8 == 0 and (BLOCK_N * hidden_dim) % (BLOCK_THREADS * 8) == 0) else DMA_ELEMS
    THREADS_PER_ROW_V = hidden_dim // VEC_V
    assert BLOCK_THREADS % THREADS_PER_ROW_V == 0
    ROWS_PER_BATCH_V = BLOCK_THREADS // THREADS_PER_ROW_V
    assert BLOCK_N % ROWS_PER_BATCH_V == 0 or ROWS_PER_BATCH_V > BLOCK_N
    NUM_BATCHES_V = max(1, BLOCK_N // ROWS_PER_BATCH_V)
    V_NEEDS_GUARD = ROWS_PER_BATCH_V > BLOCK_N

    allocator = SmemAllocator(None, global_sym_name="hstu_attention_fwd_smem")
    k_lds_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = k_lds_offset + k_lds_bytes
    v_lds_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = v_lds_offset + v_lds_bytes
    # LDS map: [K row-major tile][V row-major tile]. K is XOR-swizzled by column; V stays
    # natural [kv, d] so GEMM2 consumes it as operand B without a transpose scatter.

    # ---- Device Kernel ----
    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def hstu_attention_fwd(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        seq_offsets: fx.Tensor,
        num_targets: fx.Tensor,
        out: fx.Tensor,
    ) -> None:
        elem_type = elem_dtype.ir_type
        compute_type = fx.Float32.ir_type
        fm_fast = arith.FastMathFlags.fast
        v4f32_type = Vec.make_type(MFMA_ELEMS_PER_LANE, fx.Float32)
        mfma_pack_type = Vec.make_type(MFMA_LANE_K, elem_dtype)
        c_zero_mfma_pack = Vec.filled(MFMA_LANE_K, 0.0, elem_dtype).ir_value()

        # ---- MMA atom (layout algebra): one 16x16x16 f16/bf16 accumulate ----
        _mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(MFMA_M, MFMA_M, MFMA_K, elem_dtype))

        def mfma_acc(a_pack, b_pack, c):
            """MFMA accumulate through the layout MMA atom."""
            return fly.mma_atom_call_ssa([v4f32_type], _mma_atom, a_pack, b_pack, c)

        # ---- Thread / lane indices ----
        tid = fx.Int32(gpu.thread_idx.x)
        wave_id = tid // fx.Int32(WARP_SIZE)
        lane = tid % fx.Int32(WARP_SIZE)
        lane_mod_16 = lane % fx.Int32(MFMA_N)
        lane_div_16 = lane // fx.Int32(MFMA_N)

        # ---- Group-major grid decode -> (batch_idx, head_idx, q_tile_idx) ----
        block_id = fx.Int32(gpu.block_idx.x)
        grid_group = block_id % fx.Int32(NUM_GRID_GROUPS)
        pos_in_group = block_id // fx.Int32(NUM_GRID_GROUPS)
        local_hz_idx = pos_in_group // fx.Int32(num_q_tiles)
        q_tile_idx = pos_in_group % fx.Int32(num_q_tiles)
        hz_idx = grid_group * fx.Int32(hz_per_group) + local_hz_idx
        batch_idx = hz_idx // fx.Int32(num_heads)
        head_idx = hz_idx % fx.Int32(num_heads)

        # ---- Sequence bounds + id clamps (target tail) ----
        seq_start = fx.Int32(seq_offsets[batch_idx])
        seq_len = fx.Int32(seq_offsets[batch_idx + fx.Int32(1)]) - seq_start

        num_target = fx.Int32(0)
        if has_targets:
            num_target = fx.Int32(num_targets[batch_idx])

        # Contextual shifts max_id BEFORE the target-tail clamp (oracle _valid_attn_mask order).
        max_id = seq_len
        if has_contextual:
            max_id = seq_len - fx.Int32(contextual_seq_len) + fx.Int32(1)
        if has_targets:
            max_id = (num_target > fx.Int32(0)).select(max_id - num_target, max_id)

        # ---- Global tensor views (layout-algebra access; addresses carried by make_layout) ----
        # Group rows into g-wide coordinate slices for coalesced vector loads.
        #
        # The row*row_stride term must be computed in 64-bit: on packed tensors the largest element
        # index (L*H*dim) exceeds int32, and a 32-bit row multiply wraps -> OOB fault. A make_view
        # built from Python-int strides lowers to an i32-index view (crd2idx truncates even an i64
        # coord), so (row, head) is fixed through the tensor arg's own i64-strided layout first; the
        # resulting sub-view carries the i64 base, and the g-wide in-row vector load is a small,
        # coalesced i32 access over the < int32 in-row span.
        def grouped_loader(t, dim, g):
            in_row = fx.make_layout((dim // g, g), (g, 1))

            def load(row_i64, head_val, colgrp):
                sub = t[row_i64, head_val, None]
                return fx.make_view(fx.get_iter(sub), in_row)[colgrp, None].load()

            return load

        q_load = grouped_loader(q, head_dim, MFMA_LANE_K)
        v_load = grouped_loader(v, hidden_dim, VEC_V)  # V register-prefetch path (coalesced global -> regs)

        q_head_offset = head_idx * fx.Int32(head_dim)

        # ---- K DMA buffer resource (per-(seq,head) base folded into the descriptor) ----
        # V uses a register-prefetch path (async_load_v_regs via v_load), not a buffer resource.
        k_base_byte_offset = (fx.Int64(seq_start) * fx.Int64(stride_qk_n) + fx.Int64(q_head_offset)) * fx.Int64(2)
        k_rsrc = buffer_ops.create_buffer_resource(k, max_size=True, base_byte_offset=k_base_byte_offset)

        # LDS as shape-carried 2D views: [BLOCK_N, stride] so an LDS access is Vec.load/store on
        # (row, col) with the stride carried by the memref layout (no manual row*stride+col). The
        # buffer is still contiguous row-major (K_STRIDE has no pad, V unpadded), so the DMA base
        # extraction and contiguous buffer_load_lds fetch order are preserved.
        lds_base = allocator.get_base()
        k_smem = SmemPtr(lds_base, k_lds_offset, elem_type, shape=(BLOCK_N, K_STRIDE))
        v_smem = SmemPtr(lds_base, v_lds_offset, elem_type, shape=(BLOCK_N, V_STRIDE))
        k_lds_byte_base = buffer_ops.extract_base_index(k_smem.get(), address_space=3)

        # Single source of truth for the K LDS swizzle: XOR the column with the tile row's low bits.
        # Shared by the DMA global-fetch column and the LDS read column.
        def k_swz_col(tile_row, col):
            return col ^ ((tile_row & fx.Int32(K_SWZ_ROWS - 1)) << fx.Int32(K_SWZ_SHIFT))

        q_wave_base = q_tile_idx * fx.Int32(BLOCK_M) + wave_id * fx.Int32(ROWS_PER_WAVE)

        # ---- Q rows / bounds per query sub-tile ----
        q_rows = []
        q_in_bounds = []
        for qg in range_constexpr(Q_SUBTILES):
            local = q_wave_base + fx.Int32(qg * MFMA_M) + lane_mod_16
            q_rows.append(local)
            q_in_bounds.append(local < seq_len)

        # ---- Q B-operand packs (register-resident, per query sub-tile) ----
        q_packs = []  # q_packs[ks][qg]
        for ks in range_constexpr(K_STEPS):
            q_col = fx.Int32(ks * MFMA_K) + lane_div_16 * fx.Int32(MFMA_LANE_K)  # column within the head
            per_qg = []
            for qg in range_constexpr(Q_SUBTILES):
                safe = q_in_bounds[qg].select(seq_start + q_rows[qg], seq_start)
                raw = q_load(fx.Int64(safe), head_idx, q_col // fx.Int32(MFMA_LANE_K)).ir_value()
                per_qg.append(q_in_bounds[qg].select(raw, c_zero_mfma_pack))
            q_packs.append(per_qg)

        # ---- Score-gate helpers ----
        def _fadd(a, b):
            return arith.addf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmul(a, b):
            return arith.mulf(_raw(a), _raw(b), fastmath=fm_fast)

        c_alpha = fx.Float32(alpha)
        c_inv_n = fx.Float32(1.0 / max_seq_len)
        c_neg_log2e = fx.Float32(-_LOG2E)
        c_one_f = fx.Float32(1.0)
        c_zero_f = fx.Float32(0.0)

        def silu_scale_batch(s_list):
            """Saturating silu(alpha*s) for a list of scores, stage-batched for ILP.
            1/N is hoisted to the O epilogue."""
            sc = [_fmul(s, c_alpha) for s in s_list]
            tt = [_fmul(s, c_neg_log2e) for s in sc]
            emu = [llvm.call_intrinsic(compute_type, "llvm.amdgcn.exp2.f32", [t], [], []) for t in tt]
            den = [_fadd(c_one_f, e) for e in emu]
            sig = [llvm.call_intrinsic(compute_type, "llvm.amdgcn.rcp.f32", [d], [], []) for d in den]
            return [_fmul(sc[i], sig[i]) for i in range(len(s_list))]

        def to_id(x):
            """Raw position -> masked id (contextual prefix shift, then target-tail clamp)."""
            xid = x
            if has_contextual:
                xid = xid - fx.Int32(contextual_seq_len - 1)
                xid = (xid < fx.Int32(0)).select(fx.Int32(0), xid)
            if has_targets:
                xid = (xid > max_id).select(max_id, xid)
            return xid

        def pack_p(vals):
            """Pack 4 f32 scores into a bf16/f16 MFMA pack (the GEMM2 A-operand fragment)."""
            if is_bf16:
                c16 = fx.Int32(16)
                cmask = fx.Int32(0xFFFF0000)

                def bf16_pair(lo_f32, hi_f32):
                    lo_i32 = fx.Int32(ArithValue(lo_f32).bitcast(fx.Int32.ir_type))
                    hi_i32 = fx.Int32(ArithValue(hi_f32).bitcast(fx.Int32.ir_type))
                    return (hi_i32 & cmask) | lo_i32.shrui(c16)

                pairs = [bf16_pair(vals[0], vals[1]), bf16_pair(vals[2], vals[3])]
                return Vec.from_elements(pairs, fx.Int32).bitcast(elem_dtype).ir_value()
            elems = [fx.Float32(v).to(elem_dtype) for v in vals]
            return Vec.from_elements(elems, elem_dtype).ir_value()

        q_rows_i32 = q_rows
        q_row_ids = [to_id(q_rows_i32[qg]) for qg in range_constexpr(Q_SUBTILES)]

        # ---- KV range: causal upper bound + active predicate ----
        q_start = q_tile_idx * fx.Int32(BLOCK_M)
        q_end = q_start + fx.Int32(BLOCK_M)
        base_upper = seq_len
        if causal:
            clamped = (q_end < seq_len).select(q_end, seq_len)
            if has_contextual:
                # The prefix block holds logical row id 0, which attends the whole contextual
                # prefix (col_id < max_id) above its diagonal, so its KV range opens to seq_len.
                # Other blocks are pure causal and their high tiles are fully masked.
                ctx_block = q_start < fx.Int32(contextual_seq_len)
                base_upper = ctx_block.select(seq_len, clamped)
            else:
                base_upper = clamped
        active = q_start < seq_len
        kv_upper = active.select(base_upper, fx.Int32(0))
        n_tiles = (kv_upper + fx.Int32(BLOCK_N - 1)) // fx.Int32(BLOCK_N)

        # ---- Sliding-window lower bound: skip fully-masked low KV tiles ----
        kv_tile_start = fx.Int32(0)
        if has_window:
            eff_q_low = (q_start < max_id).select(q_start, max_id)
            kv_lower = eff_q_low - fx.Int32(max_attn_len)
            kv_lower = (kv_lower > fx.Int32(0)).select(kv_lower, fx.Int32(0))
            win_tile_start = kv_lower // fx.Int32(BLOCK_N)
            if has_contextual:
                # The prefix block must walk KV from 0 to see the prefix; the window lower bound
                # would otherwise skip the low tiles the prefix opener needs.
                ctx_prefix_block = q_start < fx.Int32(contextual_seq_len)
                kv_tile_start = ctx_prefix_block.select(fx.Int32(0), win_tile_start)
            else:
                kv_tile_start = win_tile_start

        N_ACC = D_CHUNKS * Q_SUBTILES
        c_zero_v4f32 = Vec.filled(MFMA_ELEMS_PER_LANE, 0.0, fx.Float32).ir_value()

        # ---- K DMA: global -> LDS (dword, swizzled global fetch) ----
        dma_size = fx.Int32(DMA_BYTES)
        dma_soff = fx.Int32(0)
        dma_off = fx.Int32(0)
        dma_aux = fx.Int32(1)
        c_dma_elems = fx.Int32(DMA_ELEMS)
        c_pairs_per_row_k = fx.Int32(PAIRS_PER_ROW_K)

        wave_lds_base_k = k_lds_byte_base + fx.Index(wave_id) * fx.Index(WARP_SIZE * DMA_BYTES)
        wave_lds_lane0_k = rocdl.readfirstlane(fx.Int64.ir_type, fx.Int64(wave_lds_base_k))
        k_dma_rows = []
        k_dma_gcols = []
        for d in range_constexpr(NUM_DMA_K):
            pair = tid + fx.Int32(d * BLOCK_THREADS)
            row = pair // c_pairs_per_row_k
            col_pair = pair % c_pairs_per_row_k
            col = col_pair * c_dma_elems
            k_dma_rows.append(row)
            k_dma_gcols.append(k_swz_col(row, col))

        c_stride_qk_n = fx.Int32(stride_qk_n)

        def async_load_k(kv_start):
            """Issue the async K[kv_start] dword DMA passes, global->LDS (swizzled)."""
            for d in range_constexpr(NUM_DMA_K):
                row = k_dma_rows[d]
                in_bounds = (kv_start + row) < seq_len
                local_tok = in_bounds.select(kv_start + row, fx.Int32(0))
                g_elem = local_tok * c_stride_qk_n + k_dma_gcols[d]
                voffset = g_elem * fx.Int32(2)
                lds_ptr = buffer_ops.create_llvm_ptr(wave_lds_lane0_k + fx.Int64(d * BLOCK_THREADS * DMA_BYTES), address_space=3)
                rocdl.raw_ptr_buffer_load_lds(k_rsrc, lds_ptr, dma_size, voffset, dma_soff, dma_off, dma_aux)

        # ---- V register prefetch: coalesced global -> registers (non-blocking) ----
        # Lane (tid) owns row = tid//THREADS_PER_ROW_V (+ batch*ROWS_PER_BATCH_V), col base =
        # (tid%THREADS_PER_ROW_V)*VEC_V. Reads VEC_V contiguous d elements. The load is issued but
        # NOT waited: GEMM1 runs while it is in flight, the wait deferred to a counted vmcnt(0).
        v_load_row_in_batch = tid // fx.Int32(THREADS_PER_ROW_V)
        v_load_lane_in_row = tid % fx.Int32(THREADS_PER_ROW_V)
        v_load_col = v_load_lane_in_row * fx.Int32(VEC_V)  # column within the head

        def async_load_v_regs(kv_start):
            """Issue coalesced V[kv_start] global loads to registers; return the vecs (non-blocking)."""
            vecs = []
            for b in range_constexpr(NUM_BATCHES_V):
                row = v_load_row_in_batch + fx.Int32(b * ROWS_PER_BATCH_V)
                tok = kv_start + row
                in_bounds = tok < seq_len
                if V_NEEDS_GUARD:
                    in_bounds = in_bounds & (row < fx.Int32(BLOCK_N))
                safe_tok = in_bounds.select(seq_start + tok, seq_start)
                raw = v_load(fx.Int64(safe_tok), head_idx, v_load_col // fx.Int32(VEC_V)).ir_value()
                vecs.append(in_bounds.select(raw, Vec.filled(VEC_V, 0.0, elem_dtype).ir_value()))
            return vecs

        def store_v_regs_to_lds(vecs):
            """Write prefetched V vecs to LDS row-major V[kv, d] (the GEMM2 B layout), 2D-indexed."""
            for b in range_constexpr(NUM_BATCHES_V):
                row = v_load_row_in_batch + fx.Int32(b * ROWS_PER_BATCH_V)
                Vec.store(Vec(vecs[b]), v_smem.get(), [fx.Index(row), fx.Index(v_load_col)])

        # ==== GEMM1: Q·K^T -> P (P fragment already in GEMM2 A-operand layout) ====
        def read_k_packs(ng):
            """LDS-read K A-operand packs for sub-tile ng (2D-indexed; col via the shared swizzle)."""
            k_row = fx.Int32(ng * MFMA_M) + lane_mod_16
            packs = []
            for ks in range_constexpr(K_STEPS_K):
                k_col = fx.Int32(ks * MFMA_K) + lane_div_16 * fx.Int32(MFMA_LANE_K)
                packs.append(Vec.load(mfma_pack_type, k_smem.get(), [fx.Index(k_row), fx.Index(k_swz_col(k_row, k_col))]))
            return packs

        def compute_p_tile(kv_start, k_packs_by_ng):
            """Q·K^T MFMA, apply the mask, silu-gate -> P packs for one KV tile.

            k_packs_by_ng: LDS-read A-operand packs [ng][ks] from `read_k_packs`."""
            p_packs = [[None for _ in range_constexpr(Q_SUBTILES)] for _ in range_constexpr(KV_SUBTILES)]
            for ng in range_constexpr(KV_SUBTILES):
                k_packs = [Vec(k_packs_by_ng[ng][ks]) for ks in range_constexpr(K_STEPS_K)]
                kv_base = kv_start + fx.Int32(ng * MFMA_M) + lane_div_16 * fx.Int32(MFMA_LANE_K)
                col_raw = [kv_base + fx.Int32(i) for i in range_constexpr(MFMA_ELEMS_PER_LANE)]
                col_in_seq = [col_raw[i] < seq_len for i in range_constexpr(MFMA_ELEMS_PER_LANE)]
                col_id = [to_id(col_raw[i]) for i in range_constexpr(MFMA_ELEMS_PER_LANE)]
                for qg in range_constexpr(Q_SUBTILES):
                    cur = Vec.filled(MFMA_ELEMS_PER_LANE, 0.0, fx.Float32).ir_value()
                    for ks in range_constexpr(K_STEPS_K):
                        q_op = q_packs[ks][qg] if ks < K_STEPS else c_zero_mfma_pack
                        cur = mfma_acc(k_packs[ks].ir_value(), q_op, cur)
                    s_vals = [Vec(cur)[i] for i in range_constexpr(MFMA_ELEMS_PER_LANE)]

                    def keep_col(i):
                        """causal · window · contextual · target mask for (qg, col i)."""
                        dist = q_row_ids[qg] - col_id[i]
                        keep = (q_rows_i32[qg] == col_raw[i]) | (dist > fx.Int32(0))
                        if has_window:
                            keep = keep & (dist <= fx.Int32(max_attn_len))
                        if has_contextual:
                            # Prefix opener: logical row 0 attends the whole contextual prefix.
                            ctx = (q_row_ids[qg] == fx.Int32(0)) & (col_id[i] < max_id)
                            keep = keep | ctx
                        keep = keep & col_in_seq[i]
                        return keep

                    s_vals = [keep_col(i).select(s_vals[i], c_zero_f) for i in range_constexpr(MFMA_ELEMS_PER_LANE)]
                    p_packs[ng][qg] = pack_p(silu_scale_batch(s_vals))
            return p_packs

        # ==== GEMM2: P·V -> O (V as natural-layout operand B; P as operand A) ====
        def accum_o_tile(o_acc, p_packs):
            """O[m,d] += P[m,n]·V[n,d]. A = P (GEMM1 frag, M=query K=kv), B = V[kv,d] natural."""
            for c in range_constexpr(D_CHUNKS):
                # V operand B fragment for this d-chunk: tid%16 = d, and the 4 packed elements are
                # stride-16 kv positions in row-major V[kv, d]:
                # kv = ng*16 + 4*lane_div_16 + i; d = c*16 + lane_mod_16.
                v_packs = []
                for ng in range_constexpr(KV_SUBTILES):
                    d_col = fx.Int32(c * MFMA_M) + lane_mod_16  # d index (N of GEMM2)
                    kv_lane = fx.Int32(ng * MFMA_M) + lane_div_16 * fx.Int32(MFMA_LANE_K)  # base kv
                    # b[i] = V[kv_lane + i, d_col]; 2D-indexed on the LDS view V[kv, d].
                    elems = []
                    for i in range_constexpr(MFMA_LANE_K):
                        elems.append(Vec.load(Vec.make_type(1, elem_dtype), v_smem.get(), [fx.Index(kv_lane + fx.Int32(i)), fx.Index(d_col)]))
                    v_packs.append(Vec.from_elements([Vec(e)[0] for e in elems], elem_dtype).ir_value())
                for qg in range_constexpr(Q_SUBTILES):
                    acc_off = c * Q_SUBTILES + qg
                    cur = o_acc[acc_off]
                    for ng in range_constexpr(KV_SUBTILES):
                        cur = mfma_acc(p_packs[ng][qg], v_packs[ng], cur)
                    o_acc[acc_off] = cur
            return o_acc

        # ==== Main pipeline (single K/V LDS slot, V register-prefetch overlap) ====
        # V is prefetched by ordinary global loads, so the K publish barrier does not drain it.
        # GEMM1 runs while V is in flight; vmcnt(0) drains V before the LDS publish for GEMM2.
        # Only the O accumulators are loop-carried.
        v_reg_outstanding = NUM_BATCHES_V  # V register loads kept in flight while waiting on K

        def run_kv_tile(o_acc, kv_start):
            """K staged global->LDS by swizzled DMA; V register-prefetched."""
            async_load_k(kv_start)  # K[i] global -> LDS (GEMM1 A-operand)
            v_vecs = async_load_v_regs(kv_start)  # V[i] global -> regs, in flight
            _waitcnt_vm_n(v_reg_outstanding)  # wait K[i] DMA only; V[i] stays outstanding
            gpu.barrier()  # K[i] LDS published (V[i] regs survive the barrier)
            k_packs = [read_k_packs(ng) for ng in range_constexpr(KV_SUBTILES)]
            p_packs = compute_p_tile(kv_start, k_packs)  # GEMM1(i) overlaps V[i] global load
            _waitcnt_vm_n(0)  # V[i] arrived
            store_v_regs_to_lds(v_vecs)
            rocdl.sched_group_barrier(rocdl.mask_dswr, NUM_BATCHES_V, 0)
            gpu.barrier()  # V[i] LDS published
            o_acc = accum_o_tile(o_acc, p_packs)  # GEMM2(i): O += P·V
            # The next iteration's K publish barrier also fences GEMM2(i)'s v_smem reads before
            # store_v(i+1).
            return o_acc

        if active:
            acc_init = [c_zero_v4f32 for _ in range(N_ACC)]
            loop_results = acc_init
            for kv_tile, it in range(fx.Index(kv_tile_start), fx.Index(n_tiles), fx.Index(1), init=acc_init):  # ty: ignore
                it_list = list(it) if isinstance(it, (list, tuple)) else [it]
                o_acc = [it_list[i] for i in range(N_ACC)]
                kv_start = fx.Int32(kv_tile) * fx.Int32(BLOCK_N)
                o_acc = run_kv_tile(o_acc, kv_start)
                loop_results = yield o_acc

            # ---- Epilogue: store O (1/N hoisted here) ----
            # GEMM2 writes a transposed C fragment: A=P, B=V, C[d, query].
            # tid%16 = d (N-dim), and (lane_div_16, e) = query (M-dim). The query row stored is
            # therefore q_wave_base + qg*16 + lane_div_16*4 + e; the d column is c*16 + lane_mod_16.
            results = list(loop_results) if isinstance(loop_results, (list, tuple)) else [loop_results]
            for qg in range_constexpr(Q_SUBTILES):
                q_row_base = q_wave_base + fx.Int32(qg * MFMA_M) + lane_div_16 * fx.Int32(MFMA_LANE_K)
                for e in range_constexpr(MFMA_ELEMS_PER_LANE):
                    q_row_e = q_row_base + fx.Int32(e)
                    if q_row_e < seq_len:
                        for c in range_constexpr(D_CHUNKS):
                            ov = results[c * Q_SUBTILES + qg]
                            d_col = fx.Int32(c * MFMA_M) + lane_mod_16
                            val = fx.Float32(_fmul(Vec(ov)[e], c_inv_n)).to(elem_dtype)
                            out[fx.Int64(seq_start + q_row_e), head_idx, d_col] = val

    _hstu_compile_hints = {
        "fast_fp_math": True,
        "unsafe_fp_math": True,
    }

    @flyc.jit
    def launch_hstu_attention_fwd(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        seq_offsets: fx.Tensor,
        num_targets: fx.Tensor,
        out: fx.Tensor,
        stream: fx.Stream,
    ) -> None:
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        grid = num_q_tiles * batch * num_heads
        hstu_attention_fwd(
            q,
            k,
            v,
            seq_offsets,
            num_targets,
            out,
            value_attrs={
                "passthrough": [
                    ["denormal-fp-math-f32", "preserve-sign,preserve-sign"],
                    ["no-nans-fp-math", "true"],
                    ["unsafe-fp-math", "true"],
                ],
                "rocdl.waves_per_eu": WAVES_PER_EU,
                "rocdl.flat_work_group_size": f"{BLOCK_THREADS},{BLOCK_THREADS}",
            },
        ).launch(
            grid=grid,
            block=BLOCK_THREADS,
            smem=0,
            stream=stream,
        )

    launch_hstu_attention_fwd.compile_hints = _hstu_compile_hints
    return launch_hstu_attention_fwd
