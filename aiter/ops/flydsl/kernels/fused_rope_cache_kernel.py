# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Fused RoPE + KV Cache kernel builder using the @flyc.kernel API.

Fuses 3 operations into a **single kernel launch**:
  Q -> RoPE rotation -> Q_out
  K -> RoPE rotation -> K_out + key_cache
  V -> value_cache

Grid: (max(QH, KH), T, 1)  -- shared blocks for Q and K
  block_idx.x = head_idx in [0, max(QH, KH))
  block_idx.y = token_idx

  Each block conditionally does Q work (if head_idx < QH) and/or K work
  (if head_idx < KH).  For GQA (QH >> KH) blocks beyond KH only do Q;
  for MQA-like configs where KH <= QH every block does both.

  Cos/sin are loaded ONCE per block (before branching) and shared by both
  the Q and K paths, saving buffer descriptor SGPRs.

Input shapes:
  Q: [T, QH, D],  K: [T, KH, D],  V: [T, KH, D]
  CosCache/SinCache: [max_pos, D//2] if reuse_freqs_front_part else [max_pos, D]
  Positions/SlotMapping:
    - pos_dtype="i32": [T] int32
    - pos_dtype="i64": [T] int64, accessed via stride-2 int32 indexing (.view(int32))

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

from flydsl.expr import arith, vector, buffer_ops, range_constexpr
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as _get_rocm_arch, is_rdna_arch as _is_rdna_arch

# WARP_SIZE is 32 on RDNA (wave32: gfx10xx/gfx11xx/gfx12xx) and 64 on CDNA (wave64: gfx9xx).
# All derived values (VEC_WIDTH, vecs_per_half, BLOCK_THREADS) flow from this automatically.
WARP_SIZE = 32 if _is_rdna_arch(_get_rocm_arch()) else 64


def build_fused_rope_cache_module(
    head_dim: int = 64,
    rotary_dim: int = -1,
    num_q_heads: int = 8,
    num_kv_heads: int = 1,
    block_size: int = 16,
    is_neox: bool = True,
    flash_layout: bool = True,
    dtype_str: str = "bf16",
    apply_scale: bool = False,
    reuse_freqs_front_part: bool = True,
    pos_dtype: str = "i32",
    x_size: int = 16,
):
    if rotary_dim == -1:
        rotary_dim = head_dim
    if not is_neox:
        raise NotImplementedError("Only NeoX-style RoPE is supported")
    if rotary_dim != head_dim:
        raise NotImplementedError("Partial rotation not yet supported")
    if dtype_str not in ("bf16", "f16"):
        raise ValueError(
            f"dtype_str must be 'bf16' or 'f16', got {dtype_str!r}"
        )
    half_dim = rotary_dim // 2

    # VEC_WIDTH: elements per thread. Use ceil division so vecs_per_head never
    # exceeds WARP_SIZE for the fixed one-thread-per-vector mapping below.
    # For D=64:  VEC_WIDTH=1 -> vecs_per_head=64 (full wavefront, 16-bit loads).
    # For D=96:  VEC_WIDTH=2 -> vecs_per_head=48 (fits within one wavefront).
    # For D=128: VEC_WIDTH=2 -> vecs_per_head=64 (32-bit loads, unchanged).
    VEC_WIDTH = max(1, (head_dim + WARP_SIZE - 1) // WARP_SIZE)

    vecs_per_half = half_dim // VEC_WIDTH
    vecs_per_head = head_dim // VEC_WIDTH

    # elem_bits for copy atom (bf16/f16 = 16 bits)
    elem_bits = 16
    # Copy atom bits: VEC_WIDTH * elem_bits
    copy_bits = VEC_WIDTH * elem_bits  # e.g. 2*16=32 for VEC_WIDTH=2

    if head_dim % VEC_WIDTH != 0:
        raise ValueError(f"head_dim must be a multiple of VEC_WIDTH ({VEC_WIDTH}), got {head_dim}")
    if rotary_dim % 2 != 0:
        raise ValueError(f"rotary_dim must be even, got {rotary_dim}")
    if half_dim % VEC_WIDTH != 0:
        raise ValueError(f"half_dim must be a multiple of VEC_WIDTH ({VEC_WIDTH}), got {half_dim}")
    if not flash_layout and head_dim % x_size != 0:
        raise ValueError(f"head_dim must be a multiple of x_size ({x_size}), got {head_dim}")

    BLOCK_THREADS = WARP_SIZE
    num_q_heads_val = num_q_heads
    num_kv_heads_val = num_kv_heads
    max_heads = max(num_q_heads, num_kv_heads)


    @flyc.kernel
    def fused_qk_rope_reshape_and_cache(
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
        KScale: fx.Tensor,
        VScale: fx.Tensor,
    ):
        head_idx = fx.block_idx.x
        pid_t = fx.block_idx.y
        tid = fx.thread_idx.x

        elem_type = T.bf16 if dtype_str == "bf16" else T.f16

        # --- Layout API setup ---
        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy(copy_bits), elem_bits)
        vec_reg_ty = fx.MemRefType.get(
            elem_type, fx.LayoutType.get(VEC_WIDTH, 1), fx.AddressSpace.Register
        )
        vec_reg_lay = fx.make_layout(VEC_WIDTH, 1)
        # Single layout used for both register alloca and logical_divide (same shape).
        vec_div_lay = vec_reg_lay

        # f32 scalar copy atom for KScale/VScale loads (1 x f32 = 32 bits).
        f32_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)
        f32_reg_ty = fx.MemRefType.get(T.f32, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        f32_reg_lay = fx.make_layout(1, 1)

        # Helper: load a VEC_WIDTH vector from a divided 1D tensor at given index
        def load_vec(div_tensor, idx, atom=None):
            r = fx.memref_alloca(vec_reg_ty, vec_reg_lay)
            fx.copy_atom_call(atom or copy_atom, fx.slice(div_tensor, (None, idx)), r)
            return fx.memref_load_vec(r)

        # Helper: store a VEC_WIDTH vector to a divided 1D tensor at given index
        def store_vec(val, div_tensor, idx, atom=None):
            r = fx.memref_alloca(vec_reg_ty, vec_reg_lay)
            fx.memref_store_vec(val, r)
            fx.copy_atom_call(atom or copy_atom, r, fx.slice(div_tensor, (None, idx)))

        # Helper: get the rotary-pair element via ds_bpermute (LDS cross-lane shuffle).
        # For NeoX RoPE, the pair of thread tid is tid XOR vecs_per_half.
        # ds_bpermute: thread tid reads the VGPR value held by thread (pair_byte_addr/4).
        # pair_byte_addr = (tid XOR vecs_per_half) * 4.
        # Handles VEC_WIDTH=1 (vector<1xbf16/f16>, 16-bit) and VEC_WIDTH=2 (vector<2xbf16/f16>, 32-bit).
        def ds_bpermute_pair(vec_val, pair_byte_addr):
            """Return the copy of vec_val held by the rotary-pair thread, via ds_bpermute."""
            if VEC_WIDTH == 1:
                # vector<1xf16/bf16> (16-bit) → extract scalar → bitcast to i16 → zero-extend i32
                elem_val = vector.extract(vec_val, static_position=[0], dynamic_position=[])
                i16_val = ArithValue(elem_val).bitcast(T.i16)
                i32_val = ArithValue(i16_val).extui(T.i32)
                # Cross-lane shuffle: get pair thread's 32-bit VGPR (pair elem in low 16 bits)
                peer_i32 = fx.rocdl.ds_bpermute(T.i32, pair_byte_addr, i32_val)
                # Truncate back to i16, bitcast to elem_type, reconstruct vector<1xelem_type>
                peer_i16 = ArithValue(peer_i32).trunci(T.i16)
                peer_elem = ArithValue(peer_i16).bitcast(elem_type)
                return vector.from_elements(T.vec(1, elem_type), [peer_elem])
            else:
                # VEC_WIDTH>=2: VEC_WIDTH bf16/f16 elements → n_i32 x i32, one ds_bpermute per chunk.
                # VEC_WIDTH=2 → n_i32=1 (32 bits); VEC_WIDTH=4 → n_i32=2 (64 bits), etc.
                n_i32 = VEC_WIDTH // 2
                v_i32 = vector.bitcast(T.vec(n_i32, T.i32), vec_val)
                peer_chunks = []
                for ci in range_constexpr(n_i32):
                    chunk = vector.extract(v_i32, static_position=[ci], dynamic_position=[])
                    peer_chunks.append(fx.rocdl.ds_bpermute(T.i32, pair_byte_addr, chunk))
                peer_v_i32 = vector.from_elements(T.vec(n_i32, T.i32), peer_chunks)
                return vector.bitcast(T.vec(VEC_WIDTH, elem_type), peer_v_i32)

        if tid < fx.Int32(vecs_per_head):
            # --- Load position (scalar i32) ---
            pos_rsrc = buffer_ops.create_buffer_resource(Positions, max_size=True)
            if pos_dtype == "i64":
                pos_elem_off = ArithValue(pid_t) * 2
            else:
                pos_elem_off = pid_t
            pos_val = buffer_ops.buffer_load(pos_rsrc, pos_elem_off, vec_width=1, dtype=T.i32)

            is_first_half = tid < fx.Int32(vecs_per_half)
            cos_vec_idx = tid % vecs_per_half if reuse_freqs_front_part else tid

            # Pair lane for ds_bpermute: tid XOR vecs_per_half (symmetric, works for both halves).
            # pair_byte_addr = pair_lane * 4 (ds_bpermute address unit is bytes, VGPR = 4 bytes).
            pair_lane = ArithValue(tid) ^ fx.Int32(vecs_per_half)
            pair_byte_addr = pair_lane * fx.Int32(4)

            # --- Shared cos/sin (loaded once, used by both Q and K) ---
            Cos_buf = fx.rocdl.make_buffer_tensor(CosCache)
            Sin_buf = fx.rocdl.make_buffer_tensor(SinCache)
            cos_row = fx.slice(Cos_buf, (pos_val, None))
            sin_row = fx.slice(Sin_buf, (pos_val, None))
            cos_div = fx.logical_divide(cos_row, vec_div_lay)
            sin_div = fx.logical_divide(sin_row, vec_div_lay)
            cos_e = load_vec(cos_div, cos_vec_idx)
            sin_e = load_vec(sin_div, cos_vec_idx)

            # --- Q RoPE (head_idx < num_q_heads) ---
            if head_idx < fx.Int32(num_q_heads_val):
                Q_buf = fx.rocdl.make_buffer_tensor(Q)
                Q_out_buf = fx.rocdl.make_buffer_tensor(Q_out)

                q_row = fx.slice(Q_buf, (pid_t, head_idx, None))
                q_div = fx.logical_divide(q_row, vec_div_lay)
                qo_row = fx.slice(Q_out_buf, (pid_t, head_idx, None))
                qo_div = fx.logical_divide(qo_row, vec_div_lay)

                q_e_vec = load_vec(q_div, tid)
                q_e = ArithValue(q_e_vec)
                # Use ds_bpermute to get pair element via LDS cross-lane shuffle (no VMEM).
                q_pair_e = ArithValue(ds_bpermute_pair(q_e_vec, pair_byte_addr))

                q_cos = q_e * ArithValue(cos_e)
                q_pair_sin = q_pair_e * ArithValue(sin_e)
                q_sin_term = is_first_half.select(-q_pair_sin, q_pair_sin)
                q_rot_e = q_cos + q_sin_term

                store_vec(q_rot_e.ir_value(), qo_div, tid)

            # --- K RoPE + KV cache (head_idx < num_kv_heads) ---
            if head_idx < fx.Int32(num_kv_heads_val):
                K_buf = fx.rocdl.make_buffer_tensor(K)
                K_out_buf = fx.rocdl.make_buffer_tensor(K_out)

                k_row = fx.slice(K_buf, (pid_t, head_idx, None))
                k_div = fx.logical_divide(k_row, vec_div_lay)
                ko_row = fx.slice(K_out_buf, (pid_t, head_idx, None))
                ko_div = fx.logical_divide(ko_row, vec_div_lay)

                k_e_vec = load_vec(k_div, tid)
                k_e = ArithValue(k_e_vec)
                # Use ds_bpermute to get pair element via LDS cross-lane shuffle (no VMEM).
                k_pair_e = ArithValue(ds_bpermute_pair(k_e_vec, pair_byte_addr))

                k_cos = k_e * ArithValue(cos_e)
                k_pair_sin = k_pair_e * ArithValue(sin_e)
                k_sin_term = is_first_half.select(-k_pair_sin, k_pair_sin)
                k_rot_e = k_cos + k_sin_term

                store_vec(k_rot_e.ir_value(), ko_div, tid)
                # K_buf, K_out_buf now dead — 8 SGPRs freed

                # --- KV Cache write ---
                slot_rsrc = buffer_ops.create_buffer_resource(SlotMapping, max_size=True)
                if pos_dtype == "i64":
                    slot_elem_off = ArithValue(pid_t) * 2
                else:
                    slot_elem_off = pid_t
                slot_val = buffer_ops.buffer_load(slot_rsrc, slot_elem_off, vec_width=1, dtype=T.i32)

                if slot_val >= fx.Int32(0):
                    pid_t_slot = ArithValue(slot_val) // block_size
                    pid_b = ArithValue(slot_val) % block_size

                    # Load V via layout API (deferred here to minimize SGPR liveness)
                    V_buf = fx.rocdl.make_buffer_tensor(V)
                    v_row = fx.slice(V_buf, (pid_t, head_idx, None))
                    v_div = fx.logical_divide(v_row, vec_div_lay)
                    v_e = load_vec(v_div, tid)

                    if apply_scale:
                        # --- fp8 KV cache path (raw buffer_ops for fp8 intrinsics) ---
                        ks_buf = fx.rocdl.make_buffer_tensor(KScale)
                        vs_buf = fx.rocdl.make_buffer_tensor(VScale)
                        ks_div = fx.logical_divide(fx.slice(ks_buf, (None,)), f32_reg_lay)
                        vs_div = fx.logical_divide(fx.slice(vs_buf, (None,)), f32_reg_lay)
                        r_ks = fx.memref_alloca(f32_reg_ty, f32_reg_lay)
                        r_vs = fx.memref_alloca(f32_reg_ty, f32_reg_lay)
                        fx.copy_atom_call(f32_copy_atom, fx.slice(ks_div, (None, fx.Int32(0))), r_ks)
                        fx.copy_atom_call(f32_copy_atom, fx.slice(vs_div, (None, fx.Int32(0))), r_vs)
                        k_scale_val = vector.extract(fx.memref_load_vec(r_ks), static_position=[0], dynamic_position=[])
                        v_scale_val = vector.extract(fx.memref_load_vec(r_vs), static_position=[0], dynamic_position=[])
                        k_rcp = fx.rocdl.rcp(T.f32, k_scale_val)
                        v_rcp = fx.rocdl.rcp(T.f32, v_scale_val)

                        k_scaled = []
                        v_scaled = []
                        for i in range_constexpr(VEC_WIDTH):
                            # Always use vector.extract; works for VEC_WIDTH=1 (vector<1xbf16>)
                            # and VEC_WIDTH>1 equally.
                            ke = ArithValue(vector.extract(k_rot_e.ir_value(), static_position=[i], dynamic_position=[])).extf(T.f32) * k_rcp
                            ve = ArithValue(vector.extract(v_e, static_position=[i], dynamic_position=[])).extf(T.f32) * v_rcp
                            k_scaled.append(ke)
                            v_scaled.append(ve)

                        # fp8 packing and store
                        kc_fp8_rsrc = buffer_ops.create_buffer_resource(KeyCache, max_size=True)
                        vc_fp8_rsrc = buffer_ops.create_buffer_resource(ValueCache, max_size=True)

                        if VEC_WIDTH >= 4:
                            def pack_fp8(vals):
                                i32s = []
                                for i in range_constexpr(VEC_WIDTH // 4):
                                    lo = fx.rocdl.cvt_pk_fp8_f32(
                                        T.i32, vals[i * 4], vals[i * 4 + 1], fx.Int32(0), False
                                    )
                                    wd = fx.rocdl.cvt_pk_fp8_f32(
                                        T.i32, vals[i * 4 + 2], vals[i * 4 + 3], lo, True
                                    )
                                    i32s.append(wd)
                                return i32s

                            k_fp8 = pack_fp8(k_scaled)
                            v_fp8 = pack_fp8(v_scaled)

                            if flash_layout:
                                kc_byte_off = (
                                    pid_t_slot * (block_size * num_kv_heads * head_dim)
                                    + pid_b * (num_kv_heads * head_dim)
                                    + ArithValue(head_idx) * head_dim
                                    + ArithValue(tid) * VEC_WIDTH
                                )
                                kc_dw = kc_byte_off // fx.Int32(4)
                                for wi in range_constexpr(VEC_WIDTH // 4):
                                    buffer_ops.buffer_store(k_fp8[wi], kc_fp8_rsrc, kc_dw + fx.Int32(wi))
                                    buffer_ops.buffer_store(v_fp8[wi], vc_fp8_rsrc, kc_dw + fx.Int32(wi))
                            else:
                                dim_group = ArithValue(tid) * VEC_WIDTH // x_size
                                sub_off = ArithValue(tid) * VEC_WIDTH % x_size
                                kc_byte_off = (
                                    pid_t_slot * (num_kv_heads * (head_dim // x_size) * block_size * x_size)
                                    + ArithValue(head_idx) * ((head_dim // x_size) * block_size * x_size)
                                    + dim_group * (block_size * x_size)
                                    + pid_b * x_size
                                    + sub_off
                                )
                                kc_dw = kc_byte_off // fx.Int32(4)
                                for wi in range_constexpr(VEC_WIDTH // 4):
                                    buffer_ops.buffer_store(k_fp8[wi], kc_fp8_rsrc, kc_dw + fx.Int32(wi))

                                for vi in range_constexpr(VEC_WIDTH):
                                    d_idx = ArithValue(tid) * VEC_WIDTH + vi
                                    vc_byte_off = (
                                        pid_t_slot * (num_kv_heads * head_dim * block_size)
                                        + ArithValue(head_idx) * (head_dim * block_size)
                                        + d_idx * block_size
                                        + pid_b
                                    )
                                    i32_idx = vi // 4
                                    byte_in_i32 = vi % 4
                                    shifted = ArithValue(v_fp8[i32_idx]) >> (byte_in_i32 * 8)
                                    fp8_byte = arith.trunci(T.i8, shifted)
                                    buffer_ops.buffer_store(fp8_byte, vc_fp8_rsrc, vc_byte_off)
                        else:
                            # VEC_WIDTH < 4: store individual fp8 bytes
                            for vi in range_constexpr(VEC_WIDTH):
                                k_pk = fx.rocdl.cvt_pk_fp8_f32(
                                    T.i32, k_scaled[vi], fx.Float32(0.0), fx.Int32(0), False
                                )
                                v_pk = fx.rocdl.cvt_pk_fp8_f32(
                                    T.i32, v_scaled[vi], fx.Float32(0.0), fx.Int32(0), False
                                )
                                k_byte = arith.trunci(T.i8, k_pk)
                                v_byte = arith.trunci(T.i8, v_pk)

                                d_idx = ArithValue(tid) * VEC_WIDTH + vi

                                if flash_layout:
                                    byte_off = (
                                        pid_t_slot * (block_size * num_kv_heads * head_dim)
                                        + pid_b * (num_kv_heads * head_dim)
                                        + ArithValue(head_idx) * head_dim
                                        + d_idx
                                    )
                                    buffer_ops.buffer_store(k_byte, kc_fp8_rsrc, byte_off)
                                    buffer_ops.buffer_store(v_byte, vc_fp8_rsrc, byte_off)
                                else:
                                    dim_grp = d_idx // x_size
                                    sub_o = d_idx % x_size
                                    kc_byte_off = (
                                        pid_t_slot * (num_kv_heads * (head_dim // x_size) * block_size * x_size)
                                        + ArithValue(head_idx) * ((head_dim // x_size) * block_size * x_size)
                                        + dim_grp * (block_size * x_size)
                                        + pid_b * x_size
                                        + sub_o
                                    )
                                    buffer_ops.buffer_store(k_byte, kc_fp8_rsrc, kc_byte_off)

                                    vc_byte_off = (
                                        pid_t_slot * (num_kv_heads * head_dim * block_size)
                                        + ArithValue(head_idx) * (head_dim * block_size)
                                        + d_idx * block_size
                                        + pid_b
                                    )
                                    buffer_ops.buffer_store(v_byte, vc_fp8_rsrc, vc_byte_off)
                    else:
                        # --- bf16/f16 KV cache path ---
                        if flash_layout:
                            # Flash layout: contiguous [num_blocks, block_size, KH, D]
                            KC_buf = fx.rocdl.make_buffer_tensor(KeyCache)
                            VC_buf = fx.rocdl.make_buffer_tensor(ValueCache)
                            kc_row = fx.slice(KC_buf, (pid_t_slot, pid_b, head_idx, None))
                            vc_row = fx.slice(VC_buf, (pid_t_slot, pid_b, head_idx, None))
                            kc_div = fx.logical_divide(kc_row, vec_div_lay)
                            vc_div = fx.logical_divide(vc_row, vec_div_lay)
                            store_vec(k_rot_e.ir_value(), kc_div, tid)
                            store_vec(v_e, vc_div, tid)
                        else:
                            # Non-flash layout: scattered stores, keep raw buffer_ops
                            kc_rsrc = buffer_ops.create_buffer_resource(KeyCache, max_size=True)
                            vc_rsrc = buffer_ops.create_buffer_resource(ValueCache, max_size=True)
                            for vi in range_constexpr(VEC_WIDTH):
                                d_idx = ArithValue(tid) * VEC_WIDTH + vi
                                dim_grp = d_idx // x_size
                                sub_o = d_idx % x_size
                                kc_nf_off = (
                                    pid_t_slot * (num_kv_heads * (head_dim // x_size) * block_size * x_size)
                                    + ArithValue(head_idx) * ((head_dim // x_size) * block_size * x_size)
                                    + dim_grp * (block_size * x_size)
                                    + pid_b * x_size
                                    + sub_o
                                )
                                k_elem = vector.extract(k_rot_e.ir_value(), static_position=[vi], dynamic_position=[])
                                buffer_ops.buffer_store(k_elem, kc_rsrc, kc_nf_off)

                            for vi in range_constexpr(VEC_WIDTH):
                                d_idx = ArithValue(tid) * VEC_WIDTH + vi
                                vc_nf_off = (
                                    pid_t_slot * (num_kv_heads * head_dim * block_size)
                                    + ArithValue(head_idx) * (head_dim * block_size)
                                    + d_idx * block_size
                                    + pid_b
                                )
                                v_elem = vector.extract(v_e, static_position=[vi], dynamic_position=[])
                                buffer_ops.buffer_store(v_elem, vc_rsrc, vc_nf_off)

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
        KScale: fx.Tensor,
        VScale: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = fused_qk_rope_reshape_and_cache(
            Q, K, V, Positions, CosCache, SinCache, SlotMapping,
            KeyCache, ValueCache, Q_out, K_out, KScale, VScale,
        )
        launcher.launch(
            grid=(max_heads, num_tokens, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_fused_rope_cache
