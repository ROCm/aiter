"""hstu_attention_fwd - FlyDSL kernel

out_i = (1/N) * sum_j valid(i,j) * silu(alpha * q_i * k_j^T) * v_j   (N = max_seq_len)

HSTU forward (causal / non-causal, windowed, contextual, targets; any head_dim/hidden_dim % 16).
GEMM2 consumes P as operand A and row-major V as operand B, so V stages without a transposed LDS
scatter. K stages via LDS; V loads to registers under a counted vmcnt, then publishes to LDS for
GEMM2. Row coordinates are i64 so view address arithmetic cannot overflow on large packed tensors.

Inputs/outputs:
  - q,k (L, H, head_dim); v,out (L, H, hidden_dim): packed jagged, `dtype`, rank-3 contiguous.
  - seq_offsets (Z+1) i32, num_targets (Z) i32.

Paths:
  - causal (max_attn_len == 0): causal upper bound; windowed (max_attn_len > 0) adds a
    sliding-window lower bound that skips fully-masked low KV tiles.
  - non-causal: full KV range (upper -> seq_len for every block); the per-element mask uses the
    symmetric id distance |q - col|, so max_attn_len > 0 becomes a symmetric window.
  - contextual (contextual_seq_len > 0): id shift+clamp, prefix-opener term, and (causal only)
    the prefix query block opens its KV upper bound to seq_len. In the non-causal path every
    block already spans the full KV range, so the opener needs no special KV-range handling.

Constraints:
  - gfx942 (CDNA3), gfx950 (CDNA4)
  - causal or non-causal (non-causal = full attention, or symmetric window when max_attn_len>0).
  - dtype in {f16, bf16}.
  - head_dim % 16 == 0, hidden_dim % 16 == 0;
  - fast/unsafe FP math
"""

import functools
import math as host_math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import fly
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SMEM_CAPACITY_MAP

_LOG2E = host_math.log2(host_math.e)

# s_waitcnt vmcnt encoding: vmcnt is split lo[3:0] @ bit 0, hi[5:4] @ bit 14; lgkmcnt(63) and
# expcnt(7) stay maximal so only vmcnt is constrained.
_VMCNT_LO_MASK = 0xF
_LGKMCNT_EXPCNT_BASE = 0x3F70
_VMCNT_HI_SHIFT = 14
_VMCNT_HI_MASK = 0x3


def _dtype_to_elem_type(dtype_str: str):
    if dtype_str == "f16":
        return fx.Float16
    if dtype_str == "bf16":
        return fx.BFloat16
    raise ValueError(f"unsupported dtype: {dtype_str!r} (expected 'f16' or 'bf16')")


# ---- Kernel Geometry Constants ----

WARP_SIZE = 64
# grid decoded group-major for locality
NUM_GRID_GROUPS = 8
MFMA_M = 16
MFMA_N = 16
MFMA_K = 16
MFMA_LANE_K = 4
MFMA_LANE_K_LOG2 = 2
assert (1 << MFMA_LANE_K_LOG2) == MFMA_LANE_K
MFMA_ELEMS_PER_LANE = (MFMA_M * MFMA_N) // WARP_SIZE


def _arch_dma_params(arch: str | None = None):
    """K-staging params (DMA_BYTES, DMA_ELEMS, K_SWZ_ROWS, K_SWZ_SHIFT).

    K columns are XOR-swizzled off LDS banks: swizzled_col = col ^ ((row & (ROWS-1)) << SHIFT).
    gfx942: 32 banks -> dword DMA -> (16, 2); gfx950: 64 banks -> dwordx4 DMA -> (8, 3).
    Both tile a 64-element block and the mask maxes < 64, so the XOR stays in-row (K_STRIDE % 64 == 0).
    """
    if arch is None:
        arch = get_rocm_arch()
    if (arch or "").startswith("gfx942"):
        dma_bytes, k_swz_rows, k_swz_shift = 4, 16, 2
    else:
        dma_bytes, k_swz_rows, k_swz_shift = 16, 8, 3
    return dma_bytes, dma_bytes // 2, k_swz_rows, k_swz_shift


def _waitcnt_vm_n(n: int):
    """s_waitcnt vmcnt(n) only (lgkmcnt=63, expcnt=7) so V reg loads stay outstanding across the K DMA."""
    val = (
        (n & _VMCNT_LO_MASK)
        | _LGKMCNT_EXPCNT_BASE
        | (((n >> 4) & _VMCNT_HI_MASK) << _VMCNT_HI_SHIFT)
    )
    rocdl.s_waitcnt(val)


def validate_hstu_attention_fwd(
    num_heads: int,
    head_dim: int,
    hidden_dim: int,
    causal: bool,
    max_attn_len: int,
    contextual_seq_len: int,
    has_targets: bool,
    alpha: float,
    dtype_str: str,
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
        raise ValueError(
            f"hstu attention fwd unsupported arch: {arch!r} (expected 'gfx942' or 'gfx950')"
        )

    if dtype_str not in ("f16", "bf16"):
        raise ValueError(f"unsupported dtype: {dtype_str!r} (expected 'f16' or 'bf16')")
    if contextual_seq_len < 0:
        raise ValueError(
            f"contextual_seq_len must be non-negative, got {contextual_seq_len}"
        )
    if max_attn_len < 0:
        raise ValueError(f"max_attn_len must be non-negative, got {max_attn_len}")
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}")
    if not host_math.isfinite(alpha):
        raise ValueError(f"alpha must be finite, got {alpha}")
    if head_dim <= 0 or head_dim % MFMA_K != 0:
        raise ValueError(
            f"head_dim must be positive and a multiple of MFMA_K={MFMA_K}, got {head_dim}"
        )
    if hidden_dim <= 0 or hidden_dim % MFMA_M != 0:
        raise ValueError(
            f"hidden_dim must be positive and a multiple of MFMA_M={MFMA_M}, got {hidden_dim}"
        )
    if block_m <= 0:
        raise ValueError(f"block_m must be positive, got {block_m}")
    if block_n <= 0:
        raise ValueError(f"block_n must be positive, got {block_n}")
    if num_waves <= 0:
        raise ValueError(f"num_waves must be positive, got {num_waves}")
    if waves_per_eu < 0:
        raise ValueError(f"waves_per_eu must be non-negative, got {waves_per_eu}")
    if block_m % (num_waves * MFMA_M) != 0:
        raise ValueError(
            f"block_m {block_m} must be a multiple of num_waves*MFMA_M ({num_waves * MFMA_M})"
        )
    if block_n % MFMA_M != 0:
        raise ValueError(f"block_n {block_n} must be a multiple of MFMA_M={MFMA_M}")

    _, dma_elems, _, _ = _arch_dma_params(arch)
    block_threads = num_waves * WARP_SIZE
    elems_per_dma_pass = block_threads * dma_elems
    head_dim_k = ((head_dim + 63) // 64) * 64
    if (block_n * head_dim_k) % elems_per_dma_pass != 0:
        raise ValueError("K DMA tile does not divide the DMA pass evenly")
    if (block_n * hidden_dim) % elems_per_dma_pass != 0:
        raise ValueError("V DMA tile does not divide the DMA pass evenly")

    v_dma_wide = (
        hidden_dim % 8 == 0 and (block_n * hidden_dim) % (block_threads * 8) == 0
    )
    vec_v = 8 if v_dma_wide else dma_elems
    threads_per_row_v = hidden_dim // vec_v
    if block_threads % threads_per_row_v != 0:
        raise ValueError(
            f"block_threads={block_threads} must be divisible by threads_per_row_v={threads_per_row_v}"
        )
    rows_per_batch_v = block_threads // threads_per_row_v
    if not (block_n % rows_per_batch_v == 0 or rows_per_batch_v > block_n):
        raise ValueError(
            f"rows_per_batch_v={rows_per_batch_v} must divide block_n={block_n}, unless rows_per_batch_v > block_n"
        )

    lds_cap = SMEM_CAPACITY_MAP.get(arch, 65536)
    lds_bytes = block_n * head_dim_k * 2 + block_n * hidden_dim * 2
    if lds_bytes > lds_cap:
        raise ValueError(f"LDS tile {lds_bytes} B exceeds the {lds_cap} B budget")


@functools.lru_cache(maxsize=16384)
def build_hstu_attention_fwd(
    num_heads: int,
    head_dim: int,
    hidden_dim: int,
    causal: bool,
    max_attn_len: int,
    contextual_seq_len: int,
    has_targets: bool,
    alpha: float,
    dtype_str: str,
    *,
    block_m: int = 128,
    block_n: int = 32,
    num_waves: int = 4,
    waves_per_eu: int = 2,
):
    validate_hstu_attention_fwd(
        num_heads,
        head_dim,
        hidden_dim,
        causal,
        max_attn_len,
        contextual_seq_len,
        has_targets,
        alpha,
        dtype_str,
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
    Q_SUBTILES = ROWS_PER_WAVE // MFMA_M
    KV_SUBTILES = BLOCK_N // MFMA_N
    WAVES_PER_EU = waves_per_eu

    assert num_waves > 0 and block_m % (num_waves * MFMA_M) == 0
    assert block_n % MFMA_M == 0
    assert head_dim % MFMA_K == 0
    assert hidden_dim % MFMA_M == 0

    # Arch-conditional DMA width + K LDS swizzle geometry (gfx942 dword / gfx950 dwordx4).
    DMA_BYTES, DMA_ELEMS, K_SWZ_ROWS, K_SWZ_SHIFT = _arch_dma_params()  # noqa: RUF059

    elem_dtype = _dtype_to_elem_type(dtype_str)
    is_bf16 = dtype_str == "bf16"
    has_window = max_attn_len > 0
    has_contextual = contextual_seq_len > 0

    # real 16-wide contraction steps (Q side)
    K_STEPS = head_dim // MFMA_K
    # k_swz_col has period 64, so a K_STRIDE < 64 swizzles out of row and corrupts GEMM2's A-operand.
    # Round the K stride up to 64; the extra columns over-fetch (buffer bounds -> 0) against a zero Q
    # operand and contribute nothing. head_dim % 64 == 0 leaves HEAD_DIM_K == head_dim.
    HEAD_DIM_K = ((head_dim + 63) // 64) * 64
    # Column over-fetch guard is only real when padded (HEAD_DIM_K > head_dim); otherwise the lane
    # column max is < head_dim, so it is compile-time true and dropped (else a live runtime select).
    K_COL_GUARD = head_dim < HEAD_DIM_K
    # padded contraction steps (K side); always a multiple of 4
    K_STEPS_K = HEAD_DIM_K // MFMA_K
    # GEMM2 output chunks (per-lane O accumulators)
    D_CHUNKS = hidden_dim // MFMA_M

    # K LDS tile: XOR-swizzled, K_STRIDE == HEAD_DIM_K (64-aligned so the swizzle stays in-row).
    K_STRIDE = HEAD_DIM_K

    # V LDS: store V transposed as [d, kv] so GEMM2's B-operand (4 consecutive kv for a d) is
    # contiguous -> one ds_read_b64 rather than 4x ds_read_u16. The +8 pads the [d, kv] row stride
    # to break LDS bank conflicts.
    V_T_STRIDE = BLOCK_N + 8

    # DMA pass width (gates the K/V register-load tiling below).
    elems_per_dma_pass = BLOCK_THREADS * DMA_ELEMS

    # K register-prefetch tiling: K is loaded coalesced global->registers, then stored to swizzled
    # LDS. A register destination is pipelineable across the loop backedge (the load can stay in
    # flight under a counted vmcnt while GEMM1 runs). Mirrors V's tiling.

    # 8-wide (dwordx4) load when the K tile divides an 8-wide DMA pass evenly, else the arch granule.
    k_dma_wide = (
        HEAD_DIM_K % 8 == 0 and (BLOCK_N * HEAD_DIM_K) % (BLOCK_THREADS * 8) == 0
    )
    VEC_K = 8 if k_dma_wide else DMA_ELEMS
    THREADS_PER_ROW_K = HEAD_DIM_K // VEC_K
    assert BLOCK_THREADS % THREADS_PER_ROW_K == 0
    ROWS_PER_BATCH_K = BLOCK_THREADS // THREADS_PER_ROW_K
    assert BLOCK_N % ROWS_PER_BATCH_K == 0 or ROWS_PER_BATCH_K > BLOCK_N
    NUM_BATCHES_K = max(1, BLOCK_N // ROWS_PER_BATCH_K)
    K_NEEDS_GUARD = ROWS_PER_BATCH_K > BLOCK_N
    assert VEC_K % MFMA_LANE_K == 0

    # V tile divisibility (the DMA pass also gates the register-load tiling below).
    v_tile_elems = BLOCK_N * hidden_dim
    assert v_tile_elems % elems_per_dma_pass == 0

    # V register-prefetch tiling. Each lane reads VEC_V contiguous d elements
    # of one V row, coalesced (dwordx{VEC_V//2}). One pass = ROWS_PER_BATCH_V rows; NUM_BATCHES_V
    # passes cover the BLOCK_N x hidden_dim tile. Coalesced AND overlappable (register dest, not
    # buffer_load_lds), so the load can stay in flight across GEMM1 under a counted vmcnt.

    # 8-wide (dwordx4) load when the V tile divides an 8-wide DMA pass evenly, else the arch granule.
    v_dma_wide = (
        hidden_dim % 8 == 0 and (BLOCK_N * hidden_dim) % (BLOCK_THREADS * 8) == 0
    )
    VEC_V = 8 if v_dma_wide else DMA_ELEMS
    THREADS_PER_ROW_V = hidden_dim // VEC_V
    assert BLOCK_THREADS % THREADS_PER_ROW_V == 0
    ROWS_PER_BATCH_V = BLOCK_THREADS // THREADS_PER_ROW_V
    assert BLOCK_N % ROWS_PER_BATCH_V == 0 or ROWS_PER_BATCH_V > BLOCK_N
    NUM_BATCHES_V = max(1, BLOCK_N // ROWS_PER_BATCH_V)
    V_NEEDS_GUARD = ROWS_PER_BATCH_V > BLOCK_N

    # LDS map: [K tile][V tile]. K is XOR-swizzled by column ([BLOCK_N, K_STRIDE]); V is stored
    # transposed ([hidden_dim, V_T_STRIDE]) so GEMM2's B-operand reads 4 consecutive kv per
    # ds_read_b64. Each field is a 16B-aligned fx.Array; SharedAllocator sizes the static LDS
    # global for us (no manual _align / finalize / get_base).
    @fx.struct
    class SharedStorage:
        k: fx.Array[elem_dtype, BLOCK_N * K_STRIDE, 16]
        v: fx.Array[elem_dtype, hidden_dim * V_T_STRIDE, 16]

    # ---- Device Kernel ----
    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def hstu_attention_fwd_kernel(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        seq_offsets: fx.Tensor,
        num_targets: fx.Tensor,
        out: fx.Tensor,
        num_q_tiles: fx.Int32,
        hz_per_group: fx.Int32,
        hz_total: fx.Int32,
        inv_n: fx.Float32,
    ) -> None:
        compute_type = fx.Float32.ir_type
        v4f32_type = Vec.make_type(MFMA_ELEMS_PER_LANE, fx.Float32)
        c_zero_mfma_pack = Vec.filled(MFMA_LANE_K, 0.0, elem_dtype).ir_value()

        # ---- MMA atom (layout algebra): one 16x16x16 f16/bf16 accumulate ----
        _mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(MFMA_M, MFMA_M, MFMA_K, elem_dtype))

        def mfma_acc(a_pack, b_pack, c):
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
        local_hz_idx = pos_in_group // num_q_tiles
        q_tile_idx = pos_in_group % num_q_tiles
        hz_idx = grid_group * hz_per_group + local_hz_idx

        # hz_per_group is a ceil -> the last group is padded past batch*num_heads. Padding blocks
        # clamp hz_idx=0 (in-bounds seq_offsets read) and set seq_len=0 below -> no stores, n_tiles=0.
        block_valid = hz_idx < hz_total
        hz_idx = block_valid.select(hz_idx, fx.Int32(0))
        batch_idx = hz_idx // fx.Int32(num_heads)
        head_idx = hz_idx % fx.Int32(num_heads)

        # ---- Sequence bounds + id clamps (target tail) ----
        seq_start = fx.Int32(seq_offsets[batch_idx])
        seq_len = fx.Int32(seq_offsets[batch_idx + fx.Int32(1)]) - seq_start
        seq_len = block_valid.select(seq_len, fx.Int32(0))

        num_target = fx.Int32(0)
        if has_targets:
            num_target = fx.Int32(num_targets[batch_idx])

        # Contextual shifts max_id BEFORE the target-tail clamp; to_id below applies the same order per position.
        max_id = seq_len
        if has_contextual:
            max_id = seq_len - fx.Int32(contextual_seq_len) + fx.Int32(1)
        if has_targets:
            max_id = (num_target > fx.Int32(0)).select(max_id - num_target, max_id)

        # ---- Global tensor views: g-wide coordinate slices for coalesced vector loads ----
        # (row, head) is indexed through the tensor's own i64-strided layout first (row*row_stride can
        # exceed int32 on packed tensors -> wrap/OOB); the resulting sub-view carries the i64 base and
        # the g-wide in-row load is a small i32 access over the < int32 in-row span.
        def grouped_loader(t, dim, g):
            in_row = fx.make_layout((dim // g, g), (g, 1))

            def load(row_i64, head_val, colgrp):
                sub = t[row_i64, head_val, None]
                return fx.make_view(fx.get_iter(sub), in_row)[colgrp, None].load()

            return load

        q_load = grouped_loader(q, head_dim, MFMA_LANE_K)
        v_load = grouped_loader(v, hidden_dim, VEC_V)
        k_load = grouped_loader(k, head_dim, VEC_K)

        # LDS as shape-carried views, grouped by the MFMA_LANE_K pack width so an access is
        # view[row, col_grp, None].load()/.store() (the trailing group axis carries the unit stride;
        # no manual row*stride+col). Column indices are computed directly in MFMA_LANE_K-group units
        # (see k_swz_grp and the read/store helpers) so the hot path never issues a runtime divide.
        # Views are taken once here in the enclosing region so they dominate both KV loops. k_smem is
        # K[kv, d] swizzled; v_smem is V[d, kv] transposed. Both column strides (K_STRIDE, V_T_STRIDE)
        # are MFMA_LANE_K-aligned, and every column index lands on a group boundary.
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        k_smem = lds.k.view(
            fx.make_layout(
                (BLOCK_N, K_STRIDE // MFMA_LANE_K, MFMA_LANE_K),
                (K_STRIDE, MFMA_LANE_K, 1),
            )
        )
        v_smem = lds.v.view(
            fx.make_layout(
                (hidden_dim, V_T_STRIDE // MFMA_LANE_K, MFMA_LANE_K),
                (V_T_STRIDE, MFMA_LANE_K, 1),
            )
        )

        # Single source of truth for the K LDS swizzle, expressed in MFMA_LANE_K-group units. The
        # full-column swizzle is col ^ ((row & (ROWS-1)) << SHIFT). Because MFMA_LANE_K == 1<<LOG2,
        # SHIFT >= LOG2, and every column is MFMA_LANE_K-aligned, XOR distributes over the group
        # division: (col ^ mask) // LANE == (col // LANE) ^ (mask // LANE). We therefore swizzle the
        # group index directly and index the grouped view without a runtime divide.
        assert K_SWZ_SHIFT >= MFMA_LANE_K_LOG2

        def k_swz_grp(tile_row, col_grp):
            return col_grp ^ (
                (tile_row & fx.Int32(K_SWZ_ROWS - 1))
                << fx.Int32(K_SWZ_SHIFT - MFMA_LANE_K_LOG2)
            )

        q_wave_base = q_tile_idx * fx.Int32(BLOCK_M) + wave_id * fx.Int32(ROWS_PER_WAVE)

        # ---- Q rows / bounds per query sub-tile ----
        q_rows = []
        q_in_bounds = []
        for qg in range_constexpr(Q_SUBTILES):
            local = q_wave_base + fx.Int32(qg * MFMA_M) + lane_mod_16
            q_rows.append(local)
            q_in_bounds.append(local < seq_len)

        # ---- Q B-operand packs (register-resident, per query sub-tile); q_packs[ks][qg] ----
        q_packs = []
        for ks in range_constexpr(K_STEPS):
            # column within the head
            q_col = fx.Int32(ks * MFMA_K) + lane_div_16 * fx.Int32(MFMA_LANE_K)
            per_qg = []
            for qg in range_constexpr(Q_SUBTILES):
                safe = q_in_bounds[qg].select(seq_start + q_rows[qg], seq_start)
                raw = q_load(
                    fx.Int64(safe), head_idx, q_col // fx.Int32(MFMA_LANE_K)
                ).ir_value()
                per_qg.append(q_in_bounds[qg].select(raw, c_zero_mfma_pack))
            q_packs.append(per_qg)

        # ---- Score-gate helpers ----
        # exp2(x*-log2e) == e^-x, so amdgcn.exp2/rcp build sigmoid(alpha*s) without a slow exp. Fuse
        # alpha and -log2e into one scale g = s*(alpha*-log2e): it is both the exp2 arg and the outer
        # factor p = g*sigm = (-log2e)*silu(alpha*s), so the loop does 2 muls not 3. The residual
        # -1/log2e rides the O-epilogue 1/N mul. Fast-fp comes from the launch passthrough attrs.
        c_alpha_neg_log2e = fx.Float32(alpha * -_LOG2E)
        c_one_f = fx.Float32(1.0)
        c_zero_f = fx.Float32(0.0)

        def _exp2(x):
            return fx.Float32(fx.rocdl.exp2(compute_type, x.ir_value()))

        def _rcp(x):
            return fx.Float32(fx.rocdl.rcp(compute_type, x.ir_value()))

        def silu_scale_batch(s_list):
            """silu(alpha*s) via the -log2e base change, stage-batched for ILP. Returns
            p = g*sigm with g = (alpha*-log2e)*s, so p = (-log2e)*silu(alpha*s); the -1/log2e is
            recovered in the O epilogue. A masked/zeroed score gives g=0 -> exp2(0)=1 -> sig=0.5
            -> p=0."""
            g = [s * c_alpha_neg_log2e for s in s_list]
            emu = [_exp2(gi) for gi in g]
            den = [c_one_f + e for e in emu]
            sig = [_rcp(d) for d in den]
            return [g[i] * sig[i] for i in range(len(s_list))]

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
            if const_expr(is_bf16):
                c16 = fx.Int32(16)
                cmask = fx.Int32(0xFFFF0000)

                # bf16 = the high 16 bits of each f32 (round-toward-zero truncation, no round-to-nearest).
                def bf16_pair(lo_f32, hi_f32):
                    lo_i32 = lo_f32.bitcast(fx.Int32)
                    hi_i32 = hi_f32.bitcast(fx.Int32)
                    return (hi_i32 & cmask) | lo_i32.shrui(c16)

                pairs = [bf16_pair(vals[0], vals[1]), bf16_pair(vals[2], vals[3])]
                return Vec.from_elements(pairs, fx.Int32).bitcast(elem_dtype).ir_value()
            elems = [v.to(elem_dtype) for v in vals]
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

        # ---- K register prefetch: coalesced global -> registers, then swizzled LDS store
        # (register dest is pipelineable across the backedge). Mirrors the V register path. ----
        k_load_row_in_batch = tid // fx.Int32(THREADS_PER_ROW_K)
        k_load_lane_in_row = tid % fx.Int32(THREADS_PER_ROW_K)
        # column within the padded head
        k_load_col = k_load_lane_in_row * fx.Int32(VEC_K)
        # ...same column in MFMA_LANE_K-group units (VEC_K is MFMA_LANE_K-aligned -> const multiply,
        # no runtime divide) for the swizzled LDS store below.
        k_load_col_grp = k_load_lane_in_row * fx.Int32(VEC_K // MFMA_LANE_K)

        def async_load_k_regs(kv_start, full_tile=False):
            """Issue coalesced K[kv_start] global loads to registers (non-blocking).

            full_tile (unmasked causal prefix): every row is provably in-seq, so the tok<seq_len
            bounds guard is dead -- drop it, keeping only the structural row/col guards.
            """
            vecs = []
            for b in range_constexpr(NUM_BATCHES_K):
                row = k_load_row_in_batch + fx.Int32(b * ROWS_PER_BATCH_K)
                tok = kv_start + row
                guard = None
                if const_expr(not full_tile):
                    guard = tok < seq_len
                if const_expr(K_NEEDS_GUARD):
                    rowg = row < fx.Int32(BLOCK_N)
                    guard = rowg if guard is None else (guard & rowg)
                if const_expr(K_COL_GUARD):
                    # over-fetched pad cols -> 0
                    colg = k_load_col < fx.Int32(head_dim)
                    guard = colg if guard is None else (guard & colg)
                if const_expr(guard is None):
                    raw = k_load(
                        fx.Int64(seq_start + tok),
                        head_idx,
                        k_load_col // fx.Int32(VEC_K),
                    ).ir_value()
                    vecs.append(raw)
                else:
                    safe = guard.select(seq_start + tok, seq_start)
                    raw = k_load(
                        fx.Int64(safe), head_idx, k_load_col // fx.Int32(VEC_K)
                    ).ir_value()
                    vecs.append(
                        guard.select(raw, Vec.filled(VEC_K, 0.0, elem_dtype).ir_value())
                    )
            return vecs

        def store_k_regs_to_lds(vecs):
            """Write prefetched K vecs to LDS at XOR-swizzled columns (dword-pair vector stores)."""
            for b in range_constexpr(NUM_BATCHES_K):
                row = k_load_row_in_batch + fx.Int32(b * ROWS_PER_BATCH_K)
                for h in range_constexpr(VEC_K // MFMA_LANE_K):
                    col_grp = k_load_col_grp + fx.Int32(h)
                    half = Vec.from_elements(
                        [
                            Vec(vecs[b])[h * MFMA_LANE_K + j]
                            for j in range_constexpr(MFMA_LANE_K)
                        ],
                        elem_dtype,
                    )
                    k_smem[row, k_swz_grp(row, col_grp), None].store(half)

        # ---- V register prefetch: coalesced global -> registers, issued but NOT waited so GEMM1
        # overlaps it; the wait is deferred to a counted vmcnt(0) before the V LDS publish. ----
        v_load_row_in_batch = tid // fx.Int32(THREADS_PER_ROW_V)
        v_load_lane_in_row = tid % fx.Int32(THREADS_PER_ROW_V)
        # column within the head
        v_load_col = v_load_lane_in_row * fx.Int32(VEC_V)

        def async_load_v_regs(kv_start, full_tile=False):
            """Issue coalesced V[kv_start] global loads to registers; return the vecs (non-blocking).

            full_tile=True (unmasked causal prefix): tok<seq_len is provably true (see
            async_load_k_regs) -- drop the bounds guard, keeping only the structural row guard.
            """
            vecs = []
            for b in range_constexpr(NUM_BATCHES_V):
                row = v_load_row_in_batch + fx.Int32(b * ROWS_PER_BATCH_V)
                tok = kv_start + row
                guard = None
                if const_expr(not full_tile):
                    guard = tok < seq_len
                if const_expr(V_NEEDS_GUARD):
                    # ROWS_PER_BATCH_V > BLOCK_N: one batch spans more rows than the tile, so guard
                    # row < BLOCK_N to stop surplus lanes reading past the V tile (aligned tiles: no-op).
                    rowg = row < fx.Int32(BLOCK_N)
                    guard = rowg if guard is None else (guard & rowg)
                if const_expr(guard is None):
                    raw = v_load(
                        fx.Int64(seq_start + tok),
                        head_idx,
                        v_load_col // fx.Int32(VEC_V),
                    ).ir_value()
                    vecs.append(raw)
                else:
                    safe_tok = guard.select(seq_start + tok, seq_start)
                    raw = v_load(
                        fx.Int64(safe_tok), head_idx, v_load_col // fx.Int32(VEC_V)
                    ).ir_value()
                    vecs.append(
                        guard.select(raw, Vec.filled(VEC_V, 0.0, elem_dtype).ir_value())
                    )
            return vecs

        def store_v_regs_to_lds(vecs):
            """Write prefetched V vecs to LDS, transposed to V[d, kv] so GEMM2 can read 4 consecutive
            kv per ds_read_b64. The lane's VEC_V d-contiguous elements scatter to kv-fixed, d-varying
            LDS slots."""
            for b in range_constexpr(NUM_BATCHES_V):
                kv_row = v_load_row_in_batch + fx.Int32(b * ROWS_PER_BATCH_V)
                kv_grp = kv_row // fx.Int32(MFMA_LANE_K)
                kv_lane = kv_row % fx.Int32(MFMA_LANE_K)
                vv = Vec(vecs[b])
                for j in range_constexpr(VEC_V):
                    # transpose scatter: d-contiguous lane elems land at kv-fixed, d-varying slots.
                    v_smem[v_load_col + fx.Int32(j), kv_grp, kv_lane] = vv[j]

        # ==== GEMM1: Q*K^T -> P (P fragment already in GEMM2 A-operand layout) ====
        def read_k_packs(ng):
            """LDS-read K A-operand packs for sub-tile ng (2D-indexed; col via the shared swizzle).
            The swizzle uses the LOCAL row (0..BLOCK_N-1) so it matches the store's layout.
            """
            local_k_row = fx.Int32(ng * MFMA_M) + lane_mod_16
            packs = []
            for ks in range_constexpr(K_STEPS_K):
                # k_col = ks*MFMA_K + lane_div_16*MFMA_LANE_K, in MFMA_LANE_K-group units.
                k_col_grp = fx.Int32(ks * (MFMA_K // MFMA_LANE_K)) + lane_div_16
                packs.append(
                    k_smem[local_k_row, k_swz_grp(local_k_row, k_col_grp), None].load()
                )
            return packs

        def compute_p_tile(kv_start, k_packs_by_ng, apply_mask=True):
            """Q*K^T MFMA, apply the mask, silu-gate -> P packs for one KV tile.

            apply_mask=False skips the per-element mask (and the column position/bounds math it
            needs). Only valid for a tile that is fully in-seq and fully valid -- see the causal
            unmasked-range split in the main loop. k_packs_by_ng: LDS-read A-operand packs from
            `read_k_packs`."""
            p_packs = [
                [None for _ in range_constexpr(Q_SUBTILES)]
                for _ in range_constexpr(KV_SUBTILES)
            ]
            for ng in range_constexpr(KV_SUBTILES):
                k_packs = [
                    Vec(k_packs_by_ng[ng][ks]) for ks in range_constexpr(K_STEPS_K)
                ]
                if const_expr(apply_mask):
                    kv_base = (
                        kv_start
                        + fx.Int32(ng * MFMA_M)
                        + lane_div_16 * fx.Int32(MFMA_LANE_K)
                    )
                    col_raw = [
                        kv_base + fx.Int32(i)
                        for i in range_constexpr(MFMA_ELEMS_PER_LANE)
                    ]
                    col_in_seq = [
                        col_raw[i] < seq_len
                        for i in range_constexpr(MFMA_ELEMS_PER_LANE)
                    ]
                    col_id = [
                        to_id(col_raw[i]) for i in range_constexpr(MFMA_ELEMS_PER_LANE)
                    ]
                for qg in range_constexpr(Q_SUBTILES):
                    cur = Vec.filled(MFMA_ELEMS_PER_LANE, 0.0, fx.Float32).ir_value()
                    # Operand swap: A=K, B=Q (not A=Q, B=K) so the MFMA result P lands directly in
                    # GEMM2's A-operand layout (M=query, K=kv) -- P stays register-resident, no transposed-V scatter.
                    for ks in range_constexpr(K_STEPS_K):
                        q_op = q_packs[ks][qg] if ks < K_STEPS else c_zero_mfma_pack
                        cur = mfma_acc(k_packs[ks].ir_value(), q_op, cur)
                    s_vals = [Vec(cur)[i] for i in range_constexpr(MFMA_ELEMS_PER_LANE)]

                    if const_expr(apply_mask):

                        # keep_col is traced and consumed within this same loop iteration
                        # (see the comprehension below), so the loop vars it closes over are
                        # bound at definition time -- B023's late-binding concern doesn't apply.
                        def keep_col(i):
                            """causal * window * contextual * target mask for (qg, col i)."""
                            dist = q_row_ids[qg] - col_id[i]  # noqa: B023
                            if not causal:
                                # Non-causal: symmetric id distance |q - col|. With the diagonal
                                # term this admits all in-seq columns (full attention); a window
                                # then becomes symmetric (|q - col| <= max_attn_len).
                                dist = (dist > fx.Int32(0)).select(dist, -dist)
                            # Diagonal compared in RAW positions (q_rows_i32 == col_raw): a query
                            # always attends its own token, even where to_id's shift+clamp collapses
                            # distinct ids. The window/causal distance uses to_id ids, so the id
                            # transform governs only the off-diagonal mask.
                            keep = (q_rows_i32[qg] == col_raw[i]) | (  # noqa: B023
                                dist > fx.Int32(0)
                            )
                            if has_window:
                                keep = keep & (dist <= fx.Int32(max_attn_len))
                            if has_contextual:
                                # Prefix opener: logical row 0 attends the contextual prefix.
                                ctx = (q_row_ids[qg] == fx.Int32(0)) & (  # noqa: B023
                                    col_id[i] < max_id  # noqa: B023
                                )
                                keep = keep | ctx
                            keep = keep & col_in_seq[i]  # noqa: B023
                            return keep

                        s_vals = [
                            keep_col(i).select(s_vals[i], c_zero_f)
                            for i in range_constexpr(MFMA_ELEMS_PER_LANE)
                        ]
                    p_packs[ng][qg] = pack_p(silu_scale_batch(s_vals))
            return p_packs

        # ==== GEMM2: P*V -> O (V as natural-layout operand B; P as operand A) ====
        def accum_o_tile(o_acc, p_packs):
            """O[m,d] += P[m,n]*V[n,d]. A = P (GEMM1 frag, M=query K=kv), B = V[kv,d] natural."""

            def read_v_pack(c, ng):
                # V[d, kv] transposed: 4 consecutive kv contiguous -> one ds_read_b64 = B pack.
                d_col = fx.Int32(c * MFMA_M) + lane_mod_16
                # kv_lane = ng*MFMA_M + lane_div_16*MFMA_LANE_K, in MFMA_LANE_K-group units.
                kv_grp = fx.Int32(ng * (MFMA_M // MFMA_LANE_K)) + lane_div_16
                return v_smem[d_col, kv_grp, None].load().ir_value()

            for c in range_constexpr(D_CHUNKS):
                v_packs = [read_v_pack(c, ng) for ng in range_constexpr(KV_SUBTILES)]
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
        # V register loads kept in flight while waiting on K.
        v_reg_outstanding = NUM_BATCHES_V

        def run_kv_tile(o_acc, kv_start, apply_mask=True, full_tile=False):
            """K staged global->registers->swizzled LDS (pipelineable); V register-prefetched.

            full_tile: the tile is fully in-seq (unmasked causal prefix) -> load guards elided.
            """
            k_vecs = async_load_k_regs(kv_start, full_tile=full_tile)
            v_vecs = async_load_v_regs(kv_start, full_tile=full_tile)
            # wait for K regs; V stays outstanding
            _waitcnt_vm_n(v_reg_outstanding)
            store_k_regs_to_lds(k_vecs)
            # K published to LDS
            gpu.barrier()
            k_packs = [read_k_packs(ng) for ng in range_constexpr(KV_SUBTILES)]
            # GEMM1 overlaps the in-flight V global load
            p_packs = compute_p_tile(kv_start, k_packs, apply_mask=apply_mask)
            # wait for V
            _waitcnt_vm_n(0)
            store_v_regs_to_lds(v_vecs)
            # Cluster the V ds_writes before the publish barrier (codegen-only; the barrier is the fence).
            rocdl.sched_group_barrier(rocdl.mask_dswr, NUM_BATCHES_V, 0)
            # V published to LDS
            gpu.barrier()
            # GEMM2: O += P*V. The next tile's K-publish barrier also fences these v_smem reads.
            o_acc = accum_o_tile(o_acc, p_packs)
            return o_acc

        # Tiles fully below the diagonal are entirely valid under a pure causal mask, so their
        # per-element mask is pure overhead: split into an unmasked prefix [kv_tile_start, unmasked_end)
        # and a masked remainder [unmasked_end, n_tiles). With targets, to_id clamps col > max_id to
        # max_id, so a tile is fully valid only up to min(q_start, max_id) -- the unmasked boundary.
        CAUSAL_SPLIT = causal and not has_window and not has_contextual

        if active:
            acc_init = [c_zero_v4f32 for _ in range(N_ACC)]

            # Prefix/remainder boundary. Non-causal-split -> kv_tile_start (empty prefix range).
            if const_expr(CAUSAL_SPLIT):
                split_pos = q_start
                if has_targets:
                    split_pos = (q_start < max_id).select(q_start, max_id)
                unmasked_end = split_pos // fx.Int32(BLOCK_N)
                unmasked_end = (unmasked_end > kv_tile_start).select(
                    unmasked_end, kv_tile_start
                )
                unmasked_end = (unmasked_end < n_tiles).select(unmasked_end, n_tiles)
            else:
                unmasked_end = kv_tile_start

            loop_results = acc_init
            # Unmasked prefix [kv_tile_start, unmasked_end) -- empty range unless CAUSAL_SPLIT.
            for kv_tile, it in range(
                fx.Int32(kv_tile_start),
                fx.Int32(unmasked_end),
                fx.Int32(1),
                init=acc_init,
            ):  # ty: ignore
                it_list = list(it) if isinstance(it, (list, tuple)) else [it]
                o_acc = [it_list[i] for i in range(N_ACC)]
                kv_start = fx.Int32(kv_tile) * fx.Int32(BLOCK_N)
                o_acc = run_kv_tile(o_acc, kv_start, apply_mask=False, full_tile=True)
                loop_results = yield o_acc

            # Masked remainder [unmasked_end, n_tiles).
            for kv_tile, it in range(
                fx.Int32(unmasked_end),
                fx.Int32(n_tiles),
                fx.Int32(1),
                init=loop_results,
            ):  # ty: ignore
                it_list = list(it) if isinstance(it, (list, tuple)) else [it]
                o_acc = [it_list[i] for i in range(N_ACC)]
                kv_start = fx.Int32(kv_tile) * fx.Int32(BLOCK_N)
                o_acc = run_kv_tile(o_acc, kv_start, apply_mask=True)
                loop_results = yield o_acc

            # ---- Epilogue: store O (1/N hoisted here) ----
            # GEMM2 writes a transposed C fragment: A=P, B=V, C[d, query].
            # tid%16 = d (N-dim), and (lane_div_16, e) = query (M-dim). The query row stored is
            # therefore q_wave_base + qg*16 + lane_div_16*4 + e; the d column is c*16 + lane_mod_16.
            results = (
                list(loop_results)
                if isinstance(loop_results, (list, tuple))
                else [loop_results]
            )
            for qg in range_constexpr(Q_SUBTILES):
                q_row_base = (
                    q_wave_base
                    + fx.Int32(qg * MFMA_M)
                    + lane_div_16 * fx.Int32(MFMA_LANE_K)
                )
                for e in range_constexpr(MFMA_ELEMS_PER_LANE):
                    q_row_e = q_row_base + fx.Int32(e)
                    if q_row_e < seq_len:
                        for c in range_constexpr(D_CHUNKS):
                            ov = results[c * Q_SUBTILES + qg]
                            d_col = fx.Int32(c * MFMA_M) + lane_mod_16
                            val = (Vec(ov)[e] * inv_n).to(elem_dtype)
                            out[fx.Int64(seq_start + q_row_e), head_idx, d_col] = val

    _hstu_compile_hints = {
        "fast_fp_math": True,
        "unsafe_fp_math": True,
    }

    @flyc.jit
    def launch_hstu_attention_fwd(
        max_seq_len: fx.Int32,
        batch: fx.Int32,
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        seq_offsets: fx.Tensor,
        num_targets: fx.Tensor,
        out: fx.Tensor,
        stream: fx.Stream,
    ) -> None:
        c_num_heads = fx.Int32(num_heads)
        c_ngg = fx.Int32(NUM_GRID_GROUPS)
        num_q_tiles = (max_seq_len + fx.Int32(BLOCK_M - 1)) // fx.Int32(BLOCK_M)
        hz_total = batch * c_num_heads
        hz_per_group = (hz_total + fx.Int32(NUM_GRID_GROUPS - 1)) // c_ngg
        grid_blocks = num_q_tiles * hz_per_group * c_ngg

        # Epilogue scale = (1/N)*(-1/log2e): the -1/log2e undoes silu_scale_batch's -log2e base-change
        # (alpha stays inside silu), 1/N normalizes by the global max_seq_len (N), not the row's seq_len.
        inv_n = (fx.Float32(1.0) / max_seq_len) * fx.Float32(-1.0 / _LOG2E)

        hstu_attention_fwd_kernel(
            q,
            k,
            v,
            seq_offsets,
            num_targets,
            out,
            num_q_tiles,
            hz_per_group,
            hz_total,
            inv_n,
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
            grid=(grid_blocks, 1, 1),
            block=BLOCK_THREADS,
            smem=0,
            stream=stream,
        )

    launch_hstu_attention_fwd.compile_hints = _hstu_compile_hints
    return launch_hstu_attention_fwd
