"""hstu_attention_bwd - FlyDSL KV-owned **fused** backward (dV + dK in one pass)

Backward of HSTU attention. Given dO, recompute S = alpha*Q*K^T and sigma from
Q,K (nothing is stashed by the forward), form the masked, silu-gated attention
weights, and produce both KV-owned gradient families from the single recompute:

    dV[kv, d]  = (1/N) * sum_q P[q, kv] * dO[q, d],   P  = mask .* silu(alpha*S)
    dK[kv, hc] = alpha  * sum_q dS[q, kv] * Q[q, hc], dS = mask .* (1/N) * silu'(alpha*S) .* (dO*V^T)

Both dV and dKreduce over the **query** index and share the same
S/dS fragment orientation with *no transpose*, so one program can carry both
accumulator families and compute S **once** per streamed-query tile.

Orientation (same as the forward with roles swapped): dV reduces over the query
index, so each program owns a BLOCK_M KV tile and streams BLOCK_N query tiles; K
and V are resident register operands; Q and dO are streamed through LDS. GEMM1's
C[q, kv] = S is reused as the dV/dK GEMM2 A-operand (contracting q). dV/dK rows
are single-writer.

Constraints:
  - causal + mask variants (num_targets / max_attn_len / contextual_seq_len).
  - dtype in {f16, bf16}; accumulate in fp32.
  - head_dim % 16 == 0, hidden_dim % 16 == 0; (batch*num_heads) % 8 == 0.
  - block_m must be a multiple of num_waves*16.
  - fast/unsafe FP math: not strict IEEE-754 (mirrors the forward's SiLU).
"""

import functools
import math as host_math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import fly, llvm
from flydsl._mlir.dialects.fly_rocdl import TargetAddressSpace as _TargetAddressSpace
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SMEM_CAPACITY_MAP

from aiter.ops.flydsl.kernels.hstu_attention_common import (
    decode_lane,
    grouped_loader,
    swz_col,
)


def _dtype_to_elem_type(dtype_str: str):
    if dtype_str == "f16":
        return fx.Float16
    if dtype_str == "bf16":
        return fx.BFloat16
    raise ValueError(f"unsupported dtype: {dtype_str!r} (expected 'f16' or 'bf16')")


# ---- Kernel Geometry Constants (shared with the forward) ----

WARP_SIZE = 64
NUM_GRID_GROUPS = 8
MFMA_M = 16
MFMA_N = 16
MFMA_K = 16
MFMA_LANE_K = 4
MFMA_ELEMS_PER_LANE = (MFMA_M * MFMA_N) // WARP_SIZE  # 4 f32 per lane


def _arch_dma_params(arch: str | None = None):
    if arch is None:
        arch = get_rocm_arch()
    if (arch or "").startswith("gfx942"):
        dma_bytes = 4  # CDNA3 dword
        k_swz_rows, k_swz_shift = 16, 2
    else:
        dma_bytes = 16  # CDNA4 dwordx4
        k_swz_rows, k_swz_shift = 8, 3
    return dma_bytes, dma_bytes // 2, k_swz_rows, k_swz_shift


@functools.lru_cache(maxsize=16384)
def lds_cap_bytes(arch: str | None = None) -> int:
    default_lds_cap_bytes = 65536
    if arch is None:
        arch = get_rocm_arch()
    return SMEM_CAPACITY_MAP.get(arch, default_lds_cap_bytes)


_LOG2E = host_math.log2(host_math.e)


def _waitcnt_vm_n(n: int):
    """Emit `s_waitcnt vmcnt(n)` only (lgkmcnt/expcnt left unconstrained, so
    outstanding LDS/export traffic isn't serialized).
    """
    llvm.InlineAsmOp(None, [], f"s_waitcnt vmcnt({n})", "", has_side_effects=True)


def validate_hstu_attention_bwd(
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
        raise ValueError(
            f"hstu attention bwd unsupported arch: {arch!r} (expected 'gfx942' or 'gfx950')"
        )

    if dtype_str not in {"f16", "bf16"}:
        raise ValueError(f"unsupported dtype: {dtype_str!r} (expected 'f16' or 'bf16')")
    if not causal:
        raise ValueError("hstu_attention_bwd only supports causal attention")
    if contextual_seq_len < 0:
        raise ValueError(
            f"contextual_seq_len must be non-negative, got {contextual_seq_len}"
        )
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
        raise ValueError(
            f"head_dim must be positive and a multiple of MFMA_K={MFMA_K}, got {head_dim}"
        )
    if hidden_dim <= 0 or hidden_dim % MFMA_M != 0:
        raise ValueError(
            f"hidden_dim must be positive and a multiple of MFMA_M={MFMA_M}, got {hidden_dim}"
        )
    if (batch * num_heads) % NUM_GRID_GROUPS != 0:
        raise ValueError(
            f"require (batch*num_heads) % {NUM_GRID_GROUPS} == 0, got {batch * num_heads}"
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

    _, dma_elems, _, _ = _arch_dma_params()
    block_threads = num_waves * WARP_SIZE
    elems_per_dma_pass = block_threads * dma_elems
    head_dim_k = ((head_dim + 63) // 64) * 64
    # Streamed Q tile staged through LDS: [BLOCK_N, head_dim_k].
    if (block_n * head_dim_k) % elems_per_dma_pass != 0:
        raise ValueError("Q DMA tile does not divide the dword DMA pass evenly")
    # Streamed dO tile: [BLOCK_N, hidden_dim].
    if (block_n * hidden_dim) % elems_per_dma_pass != 0:
        raise ValueError("dO DMA tile does not divide the dword DMA pass evenly")

    vec_v = (
        8
        if (hidden_dim % 8 == 0 and (block_n * hidden_dim) % (block_threads * 8) == 0)
        else dma_elems
    )
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

    lds_cap = lds_cap_bytes()
    lds_bytes = block_n * head_dim_k * 2 + block_n * hidden_dim * 2
    if lds_bytes > lds_cap:
        raise ValueError(f"LDS tile {lds_bytes} B exceeds the {lds_cap} B budget")


@functools.lru_cache(maxsize=16384)
def build_hstu_attention_bwd_dvdk(
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
    block_m: int = 64,
    block_n: int = 16,
    num_waves: int = 4,
    waves_per_eu: int = 0,
    has_perm: bool = False,
):
    validate_hstu_attention_bwd(
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
    KV_OWNED_SUBTILES = ROWS_PER_WAVE // MFMA_M
    Q_STREAM_SUBTILES = BLOCK_N // MFMA_N
    WAVES_PER_EU = waves_per_eu

    DMA_BYTES, DMA_ELEMS, K_SWZ_ROWS, K_SWZ_SHIFT = _arch_dma_params()

    elem_dtype = _dtype_to_elem_type(dtype_str)
    is_bf16 = dtype_str == "bf16"
    has_window = max_attn_len > 0
    has_contextual = contextual_seq_len > 0

    K_STEPS = head_dim // MFMA_K
    HEAD_DIM_K = ((head_dim + 63) // 64) * 64
    K_STEPS_K = HEAD_DIM_K // MFMA_K
    D_CHUNKS = hidden_dim // MFMA_M
    DK_STEPS = hidden_dim // MFMA_K
    HC_CHUNKS = head_dim // MFMA_M

    num_kv_tiles = (max_seq_len + BLOCK_M - 1) // BLOCK_M
    hz_per_group = (batch * num_heads) // NUM_GRID_GROUPS
    stride_qk_n = num_heads * head_dim

    Q_STRIDE = HEAD_DIM_K
    DO_STRIDE = hidden_dim

    q_tile_elems = BLOCK_N * Q_STRIDE
    elems_per_dma_pass = BLOCK_THREADS * DMA_ELEMS
    assert q_tile_elems % elems_per_dma_pass == 0
    NUM_DMA_Q = q_tile_elems // elems_per_dma_pass
    PAIRS_PER_ROW_Q = Q_STRIDE // DMA_ELEMS

    do_tile_elems = BLOCK_N * hidden_dim
    assert do_tile_elems % elems_per_dma_pass == 0
    # dO global->LDS DMA: row-major [q, d], no swizzle (matches the dO LDS read
    # layout). Mirrors the Q DMA sans swizzle.
    NUM_DMA_DO = do_tile_elems // elems_per_dma_pass
    PAIRS_PER_ROW_DO = DO_STRIDE // DMA_ELEMS

    N_ACC_DV = D_CHUNKS * KV_OWNED_SUBTILES
    N_ACC_DK = HC_CHUNKS * KV_OWNED_SUBTILES
    N_ACC = N_ACC_DV + N_ACC_DK

    # LDS map: [Q row-major tile][dO row-major tile]. Q is XOR-swizzled by column
    # (mirrors the forward's K tile); dO stays natural [q, d]. Each field is a
    # 16B-aligned fx.Array; SharedAllocator sizes the LDS global.
    @fx.struct
    class SharedStorage:
        q: fx.Array[elem_dtype, BLOCK_N * Q_STRIDE, 16]
        do: fx.Array[elem_dtype, BLOCK_N * DO_STRIDE, 16]

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def hstu_attention_bwd_dvdk(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        do: fx.Tensor,
        seq_offsets: fx.Tensor,
        num_targets: fx.Tensor,
        perm: fx.Tensor,
        out_dv: fx.Tensor,
        out_dk: fx.Tensor,
    ) -> None:
        elem_type = elem_dtype.ir_type
        compute_type = fx.Float32.ir_type
        v4f32_type = Vec.make_type(MFMA_ELEMS_PER_LANE, fx.Float32)
        c_zero_mfma_pack = Vec.filled(MFMA_LANE_K, 0.0, elem_dtype).ir_value()

        _mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(MFMA_M, MFMA_M, MFMA_K, elem_dtype))

        def mfma_acc(a_pack, b_pack, c):
            return fly.mma_atom_call_ssa([v4f32_type], _mma_atom, a_pack, b_pack, c)

        tid = fx.Int32(gpu.thread_idx.x)
        wave_id, lane, lane_div_16, lane_mod_16 = decode_lane(
            tid, NUM_WAVES, WARP_SIZE, MFMA_N
        )

        block_id = fx.Int32(gpu.block_idx.x)
        grid_group = block_id % fx.Int32(NUM_GRID_GROUPS)
        pos_in_group = block_id // fx.Int32(NUM_GRID_GROUPS)
        local_hz_idx = pos_in_group // fx.Int32(num_kv_tiles)
        kv_tile_idx = pos_in_group % fx.Int32(num_kv_tiles)
        hz_idx = grid_group * fx.Int32(hz_per_group) + local_hz_idx
        batch_idx = hz_idx // fx.Int32(num_heads)
        head_idx = hz_idx % fx.Int32(num_heads)

        # Group-aware sort-by-length remap (same as the split kernels): each grid
        # group processes a Sum(n^2)-balanced longest-first set. `perm` is built
        # host-side; heads stay put. The dummy 1-elem perm is never traced when off.
        if const_expr(has_perm):
            batch_idx = fx.Int32(perm[batch_idx])

        seq_start = fx.Int32(seq_offsets[batch_idx])
        seq_len = fx.Int32(seq_offsets[batch_idx + fx.Int32(1)]) - seq_start

        num_target = fx.Int32(0)
        if has_targets:
            num_target = fx.Int32(num_targets[batch_idx])
        max_id = seq_len
        if has_contextual:
            max_id = seq_len - fx.Int32(contextual_seq_len) + fx.Int32(1)
        if has_targets:
            max_id = (num_target > fx.Int32(0)).select(max_id - num_target, max_id)

        def to_id(x):
            xid = x
            if has_contextual:
                xid = xid - fx.Int32(contextual_seq_len - 1)
                xid = (xid < fx.Int32(0)).select(fx.Int32(0), xid)
            if has_targets:
                xid = (xid > max_id).select(max_id, xid)
            return xid

        k_load = grouped_loader(k, head_dim, MFMA_LANE_K)
        v_load = grouped_loader(v, hidden_dim, MFMA_LANE_K)

        q_head_offset = head_idx * fx.Int32(head_dim)
        q_base_byte_offset = (
            fx.Int64(seq_start) * fx.Int64(stride_qk_n) + fx.Int64(q_head_offset)
        ) * fx.Int64(2)

        # Shape-carried LDS views (the trailing group axis carries the stride). Q is grouped by MFMA_LANE_K for the swizzled GEMM1
        # pack read + the dK scalar gather; dO is grouped by MFMA_LANE_K for the dA
        # A-operand pack read + the dV scalar gather.
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        q_view = lds.q.view(
            fx.make_layout(
                (BLOCK_N, Q_STRIDE // MFMA_LANE_K, MFMA_LANE_K),
                (Q_STRIDE, MFMA_LANE_K, 1),
            )
        )
        do_view = lds.do.view(
            fx.make_layout(
                (BLOCK_N, DO_STRIDE // MFMA_LANE_K, MFMA_LANE_K),
                (DO_STRIDE, MFMA_LANE_K, 1),
            )
        )
        q_lds_byte_base = buffer_ops.extract_base_index(q_view, address_space=3)

        # Direct dO global->LDS DMA. dO is [L, H, hidden],
        # so the per-token stride is num_heads*hidden_dim; base at this head's slice.
        stride_do_n = num_heads * hidden_dim
        do_base_byte_offset = (
            fx.Int64(seq_start) * fx.Int64(stride_do_n)
            + fx.Int64(head_idx) * fx.Int64(hidden_dim)
        ) * fx.Int64(2)
        do_lds_byte_base = buffer_ops.extract_base_index(do_view, address_space=3)

        # ── Copy-atom global->LDS DMA (buffer_load_lds via fx.copy) ──
        # Mirrors flash_attn_gfx950's _buffer_load_lds helper: a BufferCopyLDS atom
        # drives the same buffer_load_lds instruction the raw ROCDL path did, but
        # through the FlyDSL copy-atom API (rebased buffer view + fx.copy). The atom
        # only exposes soffset/imm-offset state and hardcodes the cache-policy/aux
        # operand to 0, so this intentionally drops the raw path's aux=1.
        _buf_flags_i32 = fx.Int32(buffer_ops._get_buffer_flags())
        _dma_atom = fx.make_copy_atom(
            fx.rocdl.BufferCopyLDS(DMA_BYTES * 8), DMA_BYTES * 8
        )
        _lds_ptr_ty = fx.PointerType.get(elem_type, 2, DMA_BYTES)

        def _rebased_buffer_div(base_iter, byte_off, n_elems):
            # Fold the (large) seq/head base into the 48-bit descriptor base so the
            # per-lane element index stays a small 32-bit voffset; max_size records.
            base_i64 = fx.Int64(fx.ptrtoint(base_iter))
            shifted = fx.inttoptr(base_iter.type, base_i64 + fx.Int64(byte_off))
            buf_ptr_ty = fx.PointerType.get(
                elem_ty=elem_type,
                address_space=_TargetAddressSpace.BufferDesc,
                alignment=base_iter.alignment,
            )
            buf_ptr = fx.make_ptr(
                buf_ptr_ty,
                [
                    shifted,
                    fx.Int16(0).ir_value(),
                    fx.Int64(0xFFFFFFFF).ir_value(),
                    _buf_flags_i32.ir_value(),
                ],
            )
            return fx.logical_divide(
                fx.make_view(buf_ptr, fx.make_layout(fx.Int32(n_elems), fx.Int32(1))),
                fx.make_layout(1, 1),
            )

        q_div = _rebased_buffer_div(
            fx.get_iter(q), q_base_byte_offset, max_seq_len * stride_qk_n
        )
        do_div = _rebased_buffer_div(
            fx.get_iter(do), do_base_byte_offset, max_seq_len * stride_do_n
        )

        def q_swz_col(tile_row, col):
            return swz_col(tile_row, col, K_SWZ_ROWS, K_SWZ_SHIFT)

        kv_wave_base = kv_tile_idx * fx.Int32(BLOCK_M) + wave_id * fx.Int32(
            ROWS_PER_WAVE
        )

        kv_rows = []
        kv_in_bounds = []
        for og in range_constexpr(KV_OWNED_SUBTILES):
            local = kv_wave_base + fx.Int32(og * MFMA_M) + lane_mod_16
            kv_rows.append(local)
            kv_in_bounds.append(local < seq_len)

        k_packs = []
        for ks in range_constexpr(K_STEPS):
            k_col = fx.Int32(ks * MFMA_K) + lane_div_16 * fx.Int32(MFMA_LANE_K)
            per_og = []
            for og in range_constexpr(KV_OWNED_SUBTILES):
                safe = kv_in_bounds[og].select(seq_start + kv_rows[og], seq_start)
                raw = k_load(
                    fx.Int64(safe), head_idx, k_col // fx.Int32(MFMA_LANE_K)
                ).ir_value()
                per_og.append(kv_in_bounds[og].select(raw, c_zero_mfma_pack))
            k_packs.append(per_og)

        v_packs = []
        for ks in range_constexpr(DK_STEPS):
            v_col = fx.Int32(ks * MFMA_K) + lane_div_16 * fx.Int32(MFMA_LANE_K)
            per_og = []
            for og in range_constexpr(KV_OWNED_SUBTILES):
                safe = kv_in_bounds[og].select(seq_start + kv_rows[og], seq_start)
                raw = v_load(
                    fx.Int64(safe), head_idx, v_col // fx.Int32(MFMA_LANE_K)
                ).ir_value()
                per_og.append(kv_in_bounds[og].select(raw, c_zero_mfma_pack))
            v_packs.append(per_og)

        c_alpha = fx.Float32(alpha)
        c_inv_n = fx.Float32(1.0 / max_seq_len)
        c_neg_log2e = fx.Float32(-_LOG2E)
        c_one_f = fx.Float32(1.0)
        c_neg_one_f = fx.Float32(-1.0)
        c_zero_f = fx.Float32(0.0)

        def silu_and_grad_batch(s_list):
            # Fast (non-IEEE) SiLU + derivative on fp32 lanes. The fastmath context
            # gives every add/mul the `fast` flag (matches the compile hints). exp2
            # and rcp stay on the amdgcn approximate hardware ops: exp2 is emitted as
            # the v_exp_f32 intrinsic directly because math.exp2 lowers to a slower
            # expansion here (~1.6% on the dV/dK kernel); rcp uses the rocdl builder.
            with arith.fastmath(arith.FastMathFlags.fast):
                sc = [s * c_alpha for s in s_list]
                tt = [s * c_neg_log2e for s in sc]
                emu = [
                    fx.Float32(
                        llvm.call_intrinsic(
                            compute_type, "llvm.amdgcn.exp2.f32", [t.ir_value()], [], []
                        )
                    )
                    for t in tt
                ]
                den = [c_one_f + e for e in emu]
                sig = [fx.Float32(rocdl.rcp(compute_type, d)) for d in den]
                silu = [sc[i] * sig[i] for i in range(len(s_list))]
                grad = [
                    sig[i] * (c_one_f + sc[i] * (c_one_f + c_neg_one_f * sig[i]))
                    for i in range(len(s_list))
                ]
            return silu, grad

        def pack_p(vals):
            if is_bf16:
                c16 = fx.Int32(16)
                cmask = fx.Int32(0xFFFF0000)

                def bf16_pair(lo_f32, hi_f32):
                    lo_i32 = fx.Float32(lo_f32).bitcast(fx.Int32)
                    hi_i32 = fx.Float32(hi_f32).bitcast(fx.Int32)
                    return (hi_i32 & cmask) | lo_i32.shrui(c16)

                pairs = [bf16_pair(vals[0], vals[1]), bf16_pair(vals[2], vals[3])]
                return Vec.from_elements(pairs, fx.Int32).bitcast(elem_dtype).ir_value()
            elems = [fx.Float32(v).to(elem_dtype) for v in vals]
            return Vec.from_elements(elems, elem_dtype).ir_value()

        kv_owned_ids = [to_id(kv_rows[og]) for og in range_constexpr(KV_OWNED_SUBTILES)]

        kv_start_row = kv_tile_idx * fx.Int32(BLOCK_M)
        kv_end_row = kv_start_row + fx.Int32(BLOCK_M)
        active = kv_start_row < seq_len

        # ---- Streamed-query range: causal lower bound + optional window upper cap ----
        # Causal: queries below the KV tile never attend it (dist<=0), so start the
        # sweep at the tile's own row (contextual row-0 opener needs the full range).
        # Window: a KV row is seen only by queries within `max_attn_len` *ahead* of it
        # (KV-owned mirror of the dq window *lower* bound), so the causal `seq_len`
        # upper bound can be capped at `kv_end + max_attn_len` — the beyond-window
        # query tiles were iterated and masked to zero before (see the opt log).
        # Targets clamp to the shared id `max_id`, so a target query's raw position
        # (up to seq_len) is unrelated to its effective id: if this KV tile lies
        # within the window of `max_id` (`win_upper > max_id`), *every* target query
        # still attends and the cap must reopen to seq_len, else their dV/dK
        # contributions would be dropped. Contextual keeps the conservative seq_len
        # (prefix opener adds low-id queries that the raw-position cap can't reason
        # about); semi_local_fig has no contextual, so it takes the capped path.
        q_upper = seq_len
        if has_window and not has_contextual:
            win_upper = kv_end_row + fx.Int32(max_attn_len)
            if has_targets:
                win_upper = (win_upper <= max_id).select(win_upper, seq_len)
            q_upper = (win_upper < seq_len).select(win_upper, seq_len)
        q_upper = active.select(q_upper, fx.Int32(0))
        n_q_tiles = (q_upper + fx.Int32(BLOCK_N - 1)) // fx.Int32(BLOCK_N)
        q_tile_start = kv_start_row // fx.Int32(BLOCK_N)
        if has_contextual:
            q_tile_start = fx.Int32(0)

        c_zero_v4f32 = Vec.filled(MFMA_ELEMS_PER_LANE, 0.0, fx.Float32).ir_value()

        c_dma_elems = fx.Int32(DMA_ELEMS)
        c_pairs_per_row_q = fx.Int32(PAIRS_PER_ROW_Q)

        wave_lds_base_q = fx.Int64(q_lds_byte_base) + fx.Int64(wave_id) * fx.Int64(
            WARP_SIZE * DMA_BYTES
        )
        wave_lds_lane0_q = rocdl.readfirstlane(fx.Int64.ir_type, wave_lds_base_q)
        q_dma_rows = []
        q_dma_gcols = []
        for d in range_constexpr(NUM_DMA_Q):
            pair = tid + fx.Int32(d * BLOCK_THREADS)
            row = pair // c_pairs_per_row_q
            col_pair = pair % c_pairs_per_row_q
            col = col_pair * c_dma_elems
            q_dma_rows.append(row)
            q_dma_gcols.append(q_swz_col(row, col))

        c_stride_qk_n = fx.Int32(stride_qk_n)

        def async_load_q(q_start):
            for d in range_constexpr(NUM_DMA_Q):
                row = q_dma_rows[d]
                in_bounds = (q_start + row) < seq_len
                local_tok = in_bounds.select(q_start + row, fx.Int32(0))
                src_elem = local_tok * c_stride_qk_n + q_dma_gcols[d]
                lds_byte = fx.Int32(
                    wave_lds_lane0_q + fx.Int64(d * BLOCK_THREADS * DMA_BYTES)
                )
                dst = fx.make_view(
                    fx.inttoptr(_lds_ptr_ty, lds_byte), fx.make_layout(1, 1)
                )
                src = fx.slice(q_div, (None, fx.Int32(src_elem)))
                fx.copy(_dma_atom, src, dst)

        # Direct dO global->LDS DMA, row-major [q, d].
        # OOB q rows fetch token 0's dO (finite); their P/dS are masked to 0 so the
        # value is multiplied out — same safe-garbage contract as the Q DMA.
        c_stride_do_n = fx.Int32(stride_do_n)
        wave_lds_base_do = fx.Int64(do_lds_byte_base) + fx.Int64(wave_id) * fx.Int64(
            WARP_SIZE * DMA_BYTES
        )
        wave_lds_lane0_do = rocdl.readfirstlane(fx.Int64.ir_type, wave_lds_base_do)
        do_dma_rows = []
        do_dma_cols = []
        for d in range_constexpr(NUM_DMA_DO):
            pair = tid + fx.Int32(d * BLOCK_THREADS)
            do_dma_rows.append(pair // fx.Int32(PAIRS_PER_ROW_DO))
            do_dma_cols.append((pair % fx.Int32(PAIRS_PER_ROW_DO)) * c_dma_elems)

        def async_load_do_lds(q_start):
            for d in range_constexpr(NUM_DMA_DO):
                row = do_dma_rows[d]
                in_bounds = (q_start + row) < seq_len
                local_tok = in_bounds.select(q_start + row, fx.Int32(0))
                src_elem = local_tok * c_stride_do_n + do_dma_cols[d]
                lds_byte = fx.Int32(
                    wave_lds_lane0_do + fx.Int64(d * BLOCK_THREADS * DMA_BYTES)
                )
                dst = fx.make_view(
                    fx.inttoptr(_lds_ptr_ty, lds_byte), fx.make_layout(1, 1)
                )
                src = fx.slice(do_div, (None, fx.Int32(src_elem)))
                fx.copy(_dma_atom, src, dst)

        def read_q_packs(ng):
            q_row = fx.Int32(ng * MFMA_M) + lane_mod_16
            packs = []
            for ks in range_constexpr(K_STEPS_K):
                q_col = fx.Int32(ks * MFMA_K) + lane_div_16 * fx.Int32(MFMA_LANE_K)
                # swz_col is MFMA_LANE_K-aligned, so //MFMA_LANE_K selects the packed
                # group and the trailing group axis carries the row stride.
                packs.append(
                    q_view[
                        q_row, q_swz_col(q_row, q_col) // fx.Int32(MFMA_LANE_K), None
                    ].load()
                )
            return packs

        def compute_s_tile(q_start, q_packs_by_ng):
            p_packs = [
                [None for _ in range_constexpr(KV_OWNED_SUBTILES)]
                for _ in range_constexpr(Q_STREAM_SUBTILES)
            ]
            s_meta = [
                [None for _ in range_constexpr(KV_OWNED_SUBTILES)]
                for _ in range_constexpr(Q_STREAM_SUBTILES)
            ]
            for ng in range_constexpr(Q_STREAM_SUBTILES):
                q_packs = [
                    Vec(q_packs_by_ng[ng][ks]) for ks in range_constexpr(K_STEPS_K)
                ]
                q_base = (
                    q_start
                    + fx.Int32(ng * MFMA_M)
                    + lane_div_16 * fx.Int32(MFMA_LANE_K)
                )
                q_raw = [
                    q_base + fx.Int32(i) for i in range_constexpr(MFMA_ELEMS_PER_LANE)
                ]
                q_in_seq = [
                    q_raw[i] < seq_len for i in range_constexpr(MFMA_ELEMS_PER_LANE)
                ]
                q_ids = [to_id(q_raw[i]) for i in range_constexpr(MFMA_ELEMS_PER_LANE)]
                for og in range_constexpr(KV_OWNED_SUBTILES):
                    cur = Vec.filled(MFMA_ELEMS_PER_LANE, 0.0, fx.Float32).ir_value()
                    for ks in range_constexpr(K_STEPS_K):
                        k_op = k_packs[ks][og] if ks < K_STEPS else c_zero_mfma_pack
                        cur = mfma_acc(q_packs[ks].ir_value(), k_op, cur)
                    s_vals = [Vec(cur)[i] for i in range_constexpr(MFMA_ELEMS_PER_LANE)]

                    def keep_row(i):
                        dist = q_ids[i] - kv_owned_ids[og]
                        keep = (q_raw[i] == kv_rows[og]) | (dist > fx.Int32(0))
                        if has_window:
                            keep = keep & (dist <= fx.Int32(max_attn_len))
                        if has_contextual:
                            ctx = (q_ids[i] == fx.Int32(0)) & (
                                kv_owned_ids[og] < max_id
                            )
                            keep = keep | ctx
                        keep = keep & q_in_seq[i] & kv_in_bounds[og]
                        return keep

                    keep = [keep_row(i) for i in range_constexpr(MFMA_ELEMS_PER_LANE)]
                    silu_vals, grad_vals = silu_and_grad_batch(s_vals)
                    p_vals = [
                        keep[i].select(silu_vals[i], c_zero_f)
                        for i in range_constexpr(MFMA_ELEMS_PER_LANE)
                    ]
                    p_packs[ng][og] = pack_p(p_vals)
                    s_meta[ng][og] = (grad_vals, keep)
            return p_packs, s_meta

        def _dv_gather(c):
            # dO B-operand packs (4 adjacent q at a fixed d) for output chunk c.
            do_packs = []
            for ng in range_constexpr(Q_STREAM_SUBTILES):
                d_col = fx.Int32(c * MFMA_M) + lane_mod_16
                q_lane = fx.Int32(ng * MFMA_M) + lane_div_16 * fx.Int32(MFMA_LANE_K)
                d_grp = d_col // fx.Int32(MFMA_LANE_K)
                d_lane = d_col % fx.Int32(MFMA_LANE_K)
                elems = [
                    do_view[q_lane + fx.Int32(i), d_grp, d_lane]
                    for i in range_constexpr(MFMA_LANE_K)
                ]
                do_packs.append(Vec.from_elements(elems, elem_dtype).ir_value())
            return do_packs

        def accum_dv_tile(dv_acc, p_packs):
            # Prefetch next chunk's B-operand gather before consuming the current
            # chunk's MFMAs, so the ds_read latency overlaps the MFMA chain.
            do_cur = _dv_gather(0)
            for c in range_constexpr(D_CHUNKS):
                if const_expr(c + 1 < D_CHUNKS):
                    do_next = _dv_gather(c + 1)
                for og in range_constexpr(KV_OWNED_SUBTILES):
                    acc_off = c * KV_OWNED_SUBTILES + og
                    cur = dv_acc[acc_off]
                    for ng in range_constexpr(Q_STREAM_SUBTILES):
                        cur = mfma_acc(p_packs[ng][og], do_cur[ng], cur)
                    dv_acc[acc_off] = cur
                if const_expr(c + 1 < D_CHUNKS):
                    do_cur = do_next
            return dv_acc

        def read_do_a_packs(ng):
            q_row = fx.Int32(ng * MFMA_M) + lane_mod_16
            packs = []
            for ks in range_constexpr(DK_STEPS):
                d_col = fx.Int32(ks * MFMA_K) + lane_div_16 * fx.Int32(MFMA_LANE_K)
                packs.append(
                    do_view[q_row, d_col // fx.Int32(MFMA_LANE_K), None].load()
                )
            return packs

        def compute_ds_packs(s_meta):
            ds_packs = [
                [None for _ in range_constexpr(KV_OWNED_SUBTILES)]
                for _ in range_constexpr(Q_STREAM_SUBTILES)
            ]
            for ng in range_constexpr(Q_STREAM_SUBTILES):
                do_a = read_do_a_packs(ng)
                for og in range_constexpr(KV_OWNED_SUBTILES):
                    cur = Vec.filled(MFMA_ELEMS_PER_LANE, 0.0, fx.Float32).ir_value()
                    for ks in range_constexpr(DK_STEPS):
                        cur = mfma_acc(do_a[ks].ir_value(), v_packs[ks][og], cur)
                    da_vals = [
                        Vec(cur)[i] for i in range_constexpr(MFMA_ELEMS_PER_LANE)
                    ]
                    grad_vals, keep = s_meta[ng][og]
                    ds_vals = []
                    with arith.fastmath(arith.FastMathFlags.fast):
                        for i in range_constexpr(MFMA_ELEMS_PER_LANE):
                            gated = c_inv_n * grad_vals[i] * da_vals[i]
                            ds_vals.append(keep[i].select(gated, c_zero_f))
                    ds_packs[ng][og] = pack_p(ds_vals)
            return ds_packs

        def _dk_gather(c):
            # Q B-operand packs (4 adjacent q at a fixed hc) for output chunk c,
            # scalar-gathered from the *streamed* swizzled Q LDS view (col ->
            # group col//MFMA_LANE_K, lane col%MFMA_LANE_K), reusing GEMM1's Q — no
            # separate preshuffled q_t load.
            qb_packs = []
            for ng in range_constexpr(Q_STREAM_SUBTILES):
                hc_col = fx.Int32(c * MFMA_M) + lane_mod_16
                q_lane = fx.Int32(ng * MFMA_M) + lane_div_16 * fx.Int32(MFMA_LANE_K)
                elems = []
                for i in range_constexpr(MFMA_LANE_K):
                    q_row = q_lane + fx.Int32(i)
                    col = q_swz_col(q_row, hc_col)
                    elems.append(
                        q_view[
                            q_row,
                            col // fx.Int32(MFMA_LANE_K),
                            col % fx.Int32(MFMA_LANE_K),
                        ]
                    )
                qb_packs.append(Vec.from_elements(elems, elem_dtype).ir_value())
            return qb_packs

        def accum_dk_tile(dk_acc, ds_packs):
            qb_cur = _dk_gather(0)
            for c in range_constexpr(HC_CHUNKS):
                if const_expr(c + 1 < HC_CHUNKS):
                    qb_next = _dk_gather(c + 1)
                for og in range_constexpr(KV_OWNED_SUBTILES):
                    acc_off = c * KV_OWNED_SUBTILES + og
                    cur = dk_acc[acc_off]
                    for ng in range_constexpr(Q_STREAM_SUBTILES):
                        cur = mfma_acc(ds_packs[ng][og], qb_cur[ng], cur)
                    dk_acc[acc_off] = cur
                if const_expr(c + 1 < HC_CHUNKS):
                    qb_cur = qb_next
            return dk_acc

        def run_q_tile(acc, q_start):
            # DMA both Q and dO global->LDS, then one workgroup barrier publishes both.
            async_load_q(q_start)
            async_load_do_lds(q_start)
            _waitcnt_vm_n(0)
            gpu.barrier()
            q_packs = [read_q_packs(ng) for ng in range_constexpr(Q_STREAM_SUBTILES)]
            p_packs, s_meta = compute_s_tile(q_start, q_packs)
            dv_acc = [acc[i] for i in range(N_ACC_DV)]
            dk_acc = [acc[N_ACC_DV + i] for i in range(N_ACC_DK)]
            dv_acc = accum_dv_tile(dv_acc, p_packs)
            dk_acc = accum_dk_tile(dk_acc, compute_ds_packs(s_meta))
            return dv_acc + dk_acc

        if active:
            acc_init = [c_zero_v4f32 for _ in range(N_ACC)]
            loop_results = acc_init
            for q_tile, it in range(
                fx.Int64(q_tile_start), fx.Int64(n_q_tiles), fx.Int64(1), init=acc_init
            ):  # ty: ignore
                it_list = list(it) if isinstance(it, (list, tuple)) else [it]
                acc = [it_list[i] for i in range(N_ACC)]
                q_start = fx.Int32(q_tile) * fx.Int32(BLOCK_N)
                acc = run_q_tile(acc, q_start)
                loop_results = yield acc

            results = (
                list(loop_results)
                if isinstance(loop_results, (list, tuple))
                else [loop_results]
            )
            with arith.fastmath(arith.FastMathFlags.fast):
                for og in range_constexpr(KV_OWNED_SUBTILES):
                    kv_row_base = (
                        kv_wave_base
                        + fx.Int32(og * MFMA_M)
                        + lane_div_16 * fx.Int32(MFMA_LANE_K)
                    )
                    for e in range_constexpr(MFMA_ELEMS_PER_LANE):
                        kv_row_e = kv_row_base + fx.Int32(e)
                        if kv_row_e < seq_len:
                            for c in range_constexpr(D_CHUNKS):
                                ov = results[c * KV_OWNED_SUBTILES + og]
                                col = fx.Int32(c * MFMA_M) + lane_mod_16
                                val = (Vec(ov)[e] * c_inv_n).to(elem_dtype)
                                out_dv[
                                    fx.Int64(seq_start + kv_row_e), head_idx, col
                                ] = val
                            for c in range_constexpr(HC_CHUNKS):
                                ov = results[N_ACC_DV + c * KV_OWNED_SUBTILES + og]
                                col = fx.Int32(c * MFMA_M) + lane_mod_16
                                val = (Vec(ov)[e] * c_alpha).to(elem_dtype)
                                out_dk[
                                    fx.Int64(seq_start + kv_row_e), head_idx, col
                                ] = val

    _hstu_compile_hints = {"fast_fp_math": True, "unsafe_fp_math": True}

    @flyc.jit
    def launch_hstu_attention_bwd_dvdk(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        do: fx.Tensor,
        seq_offsets: fx.Tensor,
        num_targets: fx.Tensor,
        perm: fx.Tensor,
        out_dv: fx.Tensor,
        out_dk: fx.Tensor,
        stream: fx.Stream,
    ) -> None:
        grid = num_kv_tiles * batch * num_heads
        hstu_attention_bwd_dvdk(
            q,
            k,
            v,
            do,
            seq_offsets,
            num_targets,
            perm,
            out_dv,
            out_dk,
            value_attrs={
                "passthrough": [
                    ["denormal-fp-math-f32", "preserve-sign,preserve-sign"],
                    ["no-nans-fp-math", "true"],
                    ["unsafe-fp-math", "true"],
                ],
                "rocdl.waves_per_eu": WAVES_PER_EU,
                "rocdl.flat_work_group_size": f"{BLOCK_THREADS},{BLOCK_THREADS}",
            },
        ).launch(grid=grid, block=BLOCK_THREADS, smem=0, stream=stream)

    launch_hstu_attention_bwd_dvdk.compile_hints = _hstu_compile_hints
    return launch_hstu_attention_bwd_dvdk
