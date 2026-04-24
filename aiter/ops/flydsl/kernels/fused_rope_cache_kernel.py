# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Fused RoPE + KV Cache kernel builder using the @flyc.kernel API.

Fuses 3 operations into two kernel launches:
  Kernel 1 (Q RoPE):     Q → rotate → Q_out
  Kernel 2 (K+V cache):  K → rotate → K_out + key_cache;  V → value_cache

Input shapes:
  Q: [T, QH, D],  K: [T, KH, D],  V: [T, KH, D]
  CosCache/SinCache: [max_pos, D//2]  (must be 2-D contiguous)
  Positions: [T] int32,  SlotMapping: [T] int32

KV cache layouts:
  flash_layout=True:
    KeyCache:   [num_blocks, block_size, KH, D]
    ValueCache: [num_blocks, block_size, KH, D]
  flash_layout=False (ATOM default):
    KeyCache:   [num_blocks, KH, D//x, block_size, x]  (x=16, x-packed)
    ValueCache: [num_blocks, KH, D, block_size]         (dim-major)


"""

import flydsl.compiler as flyc
import flydsl.expr as fx

from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.numeric import Numeric
from flydsl.expr.vector import full

try:
    from kernels.kernels_common import dtype_to_elem_type
except ImportError:

    def dtype_to_elem_type(dtype_str: str):
        """Map dtype string to MLIR type (only bf16/f16 used by this kernel)."""
        if dtype_str == "bf16":
            return T.bf16
        if dtype_str == "f16":
            return T.f16
        raise ValueError(f"unsupported dtype: {dtype_str!r}")


WARP_SIZE = 64
VEC_WIDTH = 8


def build_fused_rope_cache_module(
    head_dim: int = 64,
    rotary_dim: int = -1,
    num_q_heads: int = 8,
    num_kv_heads: int = 1,
    block_size: int = 16,
    is_neox: bool = True,
    flash_layout: bool = True,
    dtype_str: str = "bf16",
):
    """Build fused RoPE + KV cache kernel.

    Args:
        head_dim: dimension per attention head
        rotary_dim: dimensions to rotate (== head_dim for full rotation)
        num_q_heads: query heads per rank
        num_kv_heads: KV heads per rank
        block_size: paged attention block size
        is_neox: True for NeoX-style rotation
        flash_layout: True for [num_blocks, block_size, KH, D] cache layout
        dtype_str: element dtype ("bf16" or "f16")

    Returns:
        launch_fn(Q, K, V, Positions, CosCache, SinCache, SlotMapping,
                  KeyCache, ValueCache, Q_out, K_out, num_tokens, stream)
    """
    if rotary_dim == -1:
        rotary_dim = head_dim
    if not is_neox:
        raise NotImplementedError("Only NeoX-style RoPE is supported")
    if rotary_dim != head_dim:
        raise NotImplementedError("Partial rotation not yet supported")
    if dtype_str not in ("bf16", "f16"):
        raise ValueError(
            f"dtype_str must be 'bf16' or 'f16', got {dtype_str!r} "
            f"(f32 is not supported: kernel uses 2-byte elem_bytes and vec8 vectorization)"
        )
    half_dim = rotary_dim // 2
    vecs_per_half = (
        half_dim // VEC_WIDTH
    )  # number of VEC_WIDTH-wide vectors covering half_dim
    vecs_per_head = (
        head_dim // VEC_WIDTH
    )  # number of VEC_WIDTH-wide vectors covering head_dim
    x_size = 16  # x-packing factor for non-flash key_cache

    # Validate vectorization and layout assumptions to avoid silent truncation.
    if head_dim % VEC_WIDTH != 0:
        raise ValueError(
            f"head_dim must be a multiple of VEC_WIDTH ({VEC_WIDTH}), "
            f"got head_dim={head_dim}"
        )
    if rotary_dim % 2 != 0:
        raise ValueError(
            f"rotary_dim must be even so that half_dim=rotary_dim//2 is integral, "
            f"got rotary_dim={rotary_dim}"
        )
    if half_dim % VEC_WIDTH != 0:
        raise ValueError(
            f"half_dim (rotary_dim//2) must be a multiple of VEC_WIDTH "
            f"({VEC_WIDTH}), got half_dim={half_dim} (rotary_dim={rotary_dim})"
        )
    if not flash_layout and head_dim % x_size != 0:
        raise ValueError(
            f"With flash_layout=False, head_dim must be a multiple of the "
            f"key_cache packing factor x_size ({x_size}), got head_dim={head_dim}"
        )
    if vecs_per_head > WARP_SIZE:
        max_head_dim = WARP_SIZE * VEC_WIDTH
        raise ValueError(
            f"Unsupported head_dim={head_dim}: with WARP_SIZE={WARP_SIZE} and "
            f"VEC_WIDTH={VEC_WIDTH}, head_dim must satisfy "
            f"head_dim <= {max_head_dim} to avoid incomplete coverage "
            f"(got vecs_per_head={vecs_per_head} > WARP_SIZE)"
        )
    BLOCK_THREADS = WARP_SIZE

    # ----- Kernel 1: Q RoPE -----
    # Grid: (T * QH, 1, 1), one program per (token, q_head)
    # Each program: vecs_per_head threads process head_dim elements
    @flyc.kernel
    def q_rope_kernel(
        Q: fx.Tensor,  # [T, QH, D]
        Positions: fx.Tensor,  # [T] int32
        CosCache: fx.Tensor,  # [max_pos, half_dim]
        SinCache: fx.Tensor,  # [max_pos, half_dim]
        Q_out: fx.Tensor,  # [T, QH, D]
    ):
        pid = fx.block_idx.x  # program id: 0..T*QH-1
        tid = fx.thread_idx.x  # 0..63

        elem_type = dtype_to_elem_type(dtype_str)
        elem_bits = 16  # bf16/f16 only

        # Buffer-backed tensors via layout API
        Q_buf = fx.rocdl.make_buffer_tensor(Q)
        Qo_buf = fx.rocdl.make_buffer_tensor(Q_out)
        Cos_buf = fx.rocdl.make_buffer_tensor(CosCache)
        Sin_buf = fx.rocdl.make_buffer_tensor(SinCache)
        Pos_buf = fx.rocdl.make_buffer_tensor(Positions)

        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
        vec_reg_ty = fx.MemRefType.get(
            elem_type, fx.LayoutType.get(VEC_WIDTH, 1), fx.AddressSpace.Register
        )
        vec_reg_lay = fx.make_layout(VEC_WIDTH, 1)

        copy_atom_i32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)
        i32_reg_ty = fx.MemRefType.get(
            T.i32, fx.LayoutType.get(1, 1), fx.AddressSpace.Register
        )
        i32_reg_lay = fx.make_layout(1, 1)

        def load_scalar_i32(buf_tensor, elem_offset):
            """Scalar i32 load using soffset for dynamic indexing."""
            div = fx.logical_divide(buf_tensor, fx.make_layout(1, 1))
            base_view = fx.slice(div, (None, fx.Int32(0)))
            atom = copy_atom_i32.set_value("soffset", elem_offset)
            r = fx.memref_alloca(i32_reg_ty, i32_reg_lay)
            fx.copy_atom_call(atom, base_view, r)
            return fx.memref_load_vec(r)[0]

        def load_vec(div_tensor, idx):
            r = fx.memref_alloca(vec_reg_ty, vec_reg_lay)
            fx.copy_atom_call(copy_atom, fx.slice(div_tensor, (None, idx)), r)
            return fx.memref_load_vec(r)

        def store_vec(val, div_tensor, idx):
            r = fx.memref_alloca(vec_reg_ty, vec_reg_lay)
            fx.memref_store_vec(val, r)
            fx.copy_atom_call(copy_atom, r, fx.slice(div_tensor, (None, idx)))

        if tid < fx.Int32(vecs_per_head):
            pid_t = pid // num_q_heads
            pid_hq = pid % num_q_heads

            pos_val = load_scalar_i32(Pos_buf, pid_t)

            # Q[pid_t, pid_hq, :] tiled by VEC_WIDTH
            q_row = fx.slice(Q_buf, (pid_t, fx.Int32(pid_hq), None))
            q_div = fx.logical_divide(q_row, fx.make_layout(VEC_WIDTH, 1))

            # Q_out[pid_t, pid_hq, :] tiled by VEC_WIDTH
            qo_row = fx.slice(Qo_buf, (pid_t, fx.Int32(pid_hq), None))
            qo_div = fx.logical_divide(qo_row, fx.make_layout(VEC_WIDTH, 1))

            # cos/sin[pos_val, :] tiled by VEC_WIDTH
            cos_row = fx.slice(Cos_buf, (pos_val, None))
            cos_div = fx.logical_divide(cos_row, fx.make_layout(VEC_WIDTH, 1))
            sin_row = fx.slice(Sin_buf, (pos_val, None))
            sin_div = fx.logical_divide(sin_row, fx.make_layout(VEC_WIDTH, 1))

            # NeoX rotation: pair with opposite half
            is_first_half = tid < fx.Int32(vecs_per_half)
            pair_tid = is_first_half.select(tid + vecs_per_half, tid - vecs_per_half)
            cos_vec_idx = tid % vecs_per_half

            qk_e = load_vec(q_div, tid)
            cos_e = load_vec(cos_div, cos_vec_idx)
            sin_e = load_vec(sin_div, cos_vec_idx)
            pair_e = load_vec(q_div, pair_tid)

            qk_cos = qk_e * cos_e
            pair_sin = pair_e * sin_e
            sin_term = is_first_half.select(-pair_sin, pair_sin)
            rot_e = qk_cos + sin_term

            store_vec(rot_e, qo_div, tid)

    # ----- Kernel 2: K RoPE + KV cache write -----
    # Grid: (T * KH, 1, 1), one program per (token, kv_head)
    # Each program: vecs_per_head threads process head_dim elements
    @flyc.kernel
    def k_cache_kernel(
        K: fx.Tensor,  # [T, KH, D]
        V: fx.Tensor,  # [T, KH, D]
        Positions: fx.Tensor,  # [T] int32
        CosCache: fx.Tensor,  # [max_pos, half_dim]
        SinCache: fx.Tensor,  # [max_pos, half_dim]
        SlotMapping: fx.Tensor,  # [T] int32
        KeyCache: fx.Tensor,  # flash: [T_cache, BS, KH, D]
        ValueCache: fx.Tensor,  # flash: [T_cache, BS, KH, D]
        K_out: fx.Tensor,  # [T, KH, D]
    ):
        pid = fx.block_idx.x  # program id: 0..T*KH-1
        tid = fx.thread_idx.x  # 0..63

        elem_type = dtype_to_elem_type(dtype_str)
        elem_dtype = Numeric.from_ir_type(elem_type)
        elem_bits = 16  # bf16/f16 only

        # Buffer-backed tensors via layout API
        K_buf = fx.rocdl.make_buffer_tensor(K)
        V_buf = fx.rocdl.make_buffer_tensor(V)
        Ko_buf = fx.rocdl.make_buffer_tensor(K_out)
        Cos_buf = fx.rocdl.make_buffer_tensor(CosCache)
        Sin_buf = fx.rocdl.make_buffer_tensor(SinCache)
        Pos_buf = fx.rocdl.make_buffer_tensor(Positions)
        Slot_buf = fx.rocdl.make_buffer_tensor(SlotMapping)
        KC_buf = fx.rocdl.make_buffer_tensor(KeyCache)
        VC_buf = fx.rocdl.make_buffer_tensor(ValueCache)

        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
        vec_reg_ty = fx.MemRefType.get(
            elem_type, fx.LayoutType.get(VEC_WIDTH, 1), fx.AddressSpace.Register
        )
        vec_reg_lay = fx.make_layout(VEC_WIDTH, 1)

        copy_atom_i32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)
        i32_reg_ty = fx.MemRefType.get(
            T.i32, fx.LayoutType.get(1, 1), fx.AddressSpace.Register
        )
        i32_reg_lay = fx.make_layout(1, 1)

        if not flash_layout:
            copy_atom_elem = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), elem_bits)
            elem_reg_ty = fx.MemRefType.get(
                elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register
            )
            elem_reg_lay = fx.make_layout(1, 1)

        def load_scalar_i32(buf_tensor, elem_offset):
            """Scalar i32 load using soffset for dynamic indexing."""
            div = fx.logical_divide(buf_tensor, fx.make_layout(1, 1))
            base_view = fx.slice(div, (None, fx.Int32(0)))
            atom = copy_atom_i32.set_value("soffset", elem_offset)
            r = fx.memref_alloca(i32_reg_ty, i32_reg_lay)
            fx.copy_atom_call(atom, base_view, r)
            return fx.memref_load_vec(r)[0]

        def load_vec(div_tensor, idx):
            r = fx.memref_alloca(vec_reg_ty, vec_reg_lay)
            fx.copy_atom_call(copy_atom, fx.slice(div_tensor, (None, idx)), r)
            return fx.memref_load_vec(r)

        def store_vec(val, div_tensor, idx):
            r = fx.memref_alloca(vec_reg_ty, vec_reg_lay)
            fx.memref_store_vec(val, r)
            fx.copy_atom_call(copy_atom, r, fx.slice(div_tensor, (None, idx)))

        def store_scalar(val, div_tensor, idx):
            r = fx.memref_alloca(elem_reg_ty, elem_reg_lay)
            ts = full(1, elem_dtype(val), elem_dtype)
            fx.memref_store_vec(ts, r)
            fx.copy_atom_call(copy_atom_elem, r, fx.slice(div_tensor, (None, idx)))

        if tid < fx.Int32(vecs_per_head):
            pid_t = pid // num_kv_heads
            pid_hk = pid % num_kv_heads

            pos_val = load_scalar_i32(Pos_buf, pid_t)

            # K[pid_t, pid_hk, :] tiled by VEC_WIDTH
            k_row = fx.slice(K_buf, (pid_t, fx.Int32(pid_hk), None))
            k_div = fx.logical_divide(k_row, fx.make_layout(VEC_WIDTH, 1))

            # K_out[pid_t, pid_hk, :] tiled by VEC_WIDTH
            ko_row = fx.slice(Ko_buf, (pid_t, fx.Int32(pid_hk), None))
            ko_div = fx.logical_divide(ko_row, fx.make_layout(VEC_WIDTH, 1))

            # cos/sin[pos_val, :] tiled by VEC_WIDTH
            cos_row = fx.slice(Cos_buf, (pos_val, None))
            cos_div = fx.logical_divide(cos_row, fx.make_layout(VEC_WIDTH, 1))
            sin_row = fx.slice(Sin_buf, (pos_val, None))
            sin_div = fx.logical_divide(sin_row, fx.make_layout(VEC_WIDTH, 1))

            # NeoX rotation
            is_first_half = tid < fx.Int32(vecs_per_half)
            pair_tid = is_first_half.select(tid + vecs_per_half, tid - vecs_per_half)
            cos_vec_idx = tid % vecs_per_half

            qk_e = load_vec(k_div, tid)
            cos_e = load_vec(cos_div, cos_vec_idx)
            sin_e = load_vec(sin_div, cos_vec_idx)
            pair_e = load_vec(k_div, pair_tid)

            qk_cos = qk_e * cos_e
            pair_sin = pair_e * sin_e
            sin_term = is_first_half.select(-pair_sin, pair_sin)
            k_rot_e = qk_cos + sin_term

            store_vec(k_rot_e, ko_div, tid)

            # --- KV Cache write ---
            slot_val = load_scalar_i32(Slot_buf, pid_t)

            if slot_val >= fx.Int32(0):
                pid_t_slot = slot_val // block_size
                pid_b = slot_val % block_size

                # Load V
                v_row = fx.slice(V_buf, (pid_t, fx.Int32(pid_hk), None))
                v_div = fx.logical_divide(v_row, fx.make_layout(VEC_WIDTH, 1))
                v_e = load_vec(v_div, tid)

                if flash_layout:
                    # Flash: [num_blocks, block_size, KH, D] → 1D, tile by VEC_WIDTH
                    kc_row = fx.slice(
                        KC_buf, (pid_t_slot, pid_b, fx.Int32(pid_hk), None)
                    )
                    kc_div = fx.logical_divide(kc_row, fx.make_layout(VEC_WIDTH, 1))
                    vc_row = fx.slice(
                        VC_buf, (pid_t_slot, pid_b, fx.Int32(pid_hk), None)
                    )
                    vc_div = fx.logical_divide(vc_row, fx.make_layout(VEC_WIDTH, 1))

                    store_vec(k_rot_e, kc_div, tid)
                    store_vec(v_e, vc_div, tid)
                else:
                    # Non-flash key_cache: [num_blocks, KH, D//x, BS, x]
                    dim_group = (tid * VEC_WIDTH) // x_size
                    sub_tile = tid % (x_size // VEC_WIDTH)

                    kc_nf_row = fx.slice(
                        KC_buf, (pid_t_slot, fx.Int32(pid_hk), dim_group, pid_b, None)
                    )
                    kc_nf_div = fx.logical_divide(
                        kc_nf_row, fx.make_layout(VEC_WIDTH, 1)
                    )
                    store_vec(k_rot_e, kc_nf_div, sub_tile)

                    # Non-flash value_cache: [num_blocks, KH, D, block_size]
                    for vi in range_constexpr(VEC_WIDTH):
                        v_scalar = v_e[vi]
                        d_idx = tid * VEC_WIDTH + vi
                        vc_row = fx.slice(
                            VC_buf, (pid_t_slot, fx.Int32(pid_hk), d_idx, None)
                        )
                        vc_div = fx.logical_divide(vc_row, fx.make_layout(1, 1))
                        store_scalar(v_scalar, vc_div, pid_b)

    @flyc.jit
    def launch_fused_rope_cache(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        Positions: fx.Tensor,
        CosCache: fx.Tensor,
        SinCache: fx.Tensor,
        SlotMapping: fx.Tensor,
        KeyCache: fx.Tensor,
        ValueCache: fx.Tensor,
        Q_out: fx.Tensor,
        K_out: fx.Tensor,
        num_tokens: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        # Kernel 1: Q RoPE
        n_q = num_tokens * num_q_heads
        q_launcher = q_rope_kernel(Q, Positions, CosCache, SinCache, Q_out)
        q_launcher.launch(
            grid=(n_q, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

        # Kernel 2: K RoPE + KV cache write
        n_k = num_tokens * num_kv_heads
        k_launcher = k_cache_kernel(
            K,
            V,
            Positions,
            CosCache,
            SinCache,
            SlotMapping,
            KeyCache,
            ValueCache,
            K_out,
        )
        k_launcher.launch(
            grid=(n_k, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_fused_rope_cache
