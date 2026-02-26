"""Shared MFMA preshuffle helpers (used by preshuffle GEMM + MoE kernels).

This module consolidates the common building blocks that were previously duplicated
across:
- `kernels/preshuffle_gemm.py`
- `kernels/moe_gemm_2stage.py`

Key primitives:
- B preshuffle layout builder (supports byte-packed element types, incl. packed int4)
- B pack load for MFMA K32 micro-steps (8B output pack; optional int4->int8 unpack)
"""

from __future__ import annotations
from dataclasses import dataclass
from _mlir import ir

@dataclass(frozen=True)
class PreshuffleBLayout:
    """Container returned by `make_preshuffle_b_layout`."""

    layout_b: object
    kpack_bytes: int


def make_preshuffle_b_layout(
    flir,
    arith,
    *,
    c_n: ir.Value,
    c_k: ir.Value,
    kpack_bytes: int = 16,
    elem_bytes: int = 1,
) -> PreshuffleBLayout:
    """Build B layout matching aiter/CK preshuffle for A8 MFMA kernels.

    Shape: (N0, K0, KLane, NLane, KPackBytes) = (N/16, K/64, 4, 16, kpack_bytes)

    Notes:
    - For FP8/INT8: kpack_bytes=16 (one byte per element).
    - For packed INT4 (W4A8): kpack_bytes=8 (two 4-bit values per byte).
    """
    if kpack_bytes not in (8, 16):
        raise ValueError(f"kpack_bytes must be 8 or 16, got {kpack_bytes!r}")

    c16 = arith.constant(16, index=True)
    c64 = arith.constant(64, index=True)
    c4 = arith.constant(4, index=True)
    c_kpack = arith.constant(kpack_bytes, index=True)

    # This layout is fundamentally byte-addressed along K:
    # - For 1B types (fp8/i8): KBytes == K
    # - For 2B types (fp16/bf16): KBytes == 2*K
    #
    # We keep the same 64B K0 "macro-step" used by CK/aiter preshuffle.
    if elem_bytes not in (1, 2):
        raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")
    c_k_bytes = c_k * arith.constant(int(elem_bytes), index=True)
    c_k0 = c_k_bytes / c64
    n0 = c_n / c16

    # Layout is expressed in ELEMENT units (not bytes). Convert KPackBytes -> KPackElems.
    c_kpack_elems = c_kpack if elem_bytes == 1 else (c_kpack / arith.constant(int(elem_bytes), index=True))

    # Strides derived from the layout shape:
    # - KPack stride = 1
    # - NLane stride = KPackElems
    # - KLane stride = NLane * KPackElems = 16 * KPackElems
    # - K0   stride = KLane * NLane * KPackElems = 4 * 16 * KPackElems
    stride_nlane = c_kpack_elems
    stride_klane = c16 * stride_nlane
    stride_k0 = c4 * stride_klane
    stride_n0 = c_k0 * stride_k0

    stride_b = (
        stride_n0,      # n0
        stride_k0,      # k0
        stride_klane,   # k1 (KLane)
        stride_nlane,   # n1
        arith.constant(1, index=True),  # k2
    )
    layout_b = flir.make_layout((n0, c_k0, c4, c16, c_kpack_elems), stride=stride_b)
    return PreshuffleBLayout(layout_b=layout_b, kpack_bytes=kpack_bytes)


def make_preshuffle_scale_layout(
    flir,
    arith,
    *,
    c_mn: ir.Value,
    c_k: ir.Value,
    mn_pack: int = 2,
    k_pack: int = 2,
    elem_bytes: int = 4,
    scale_block_size: int = 32,
) -> object:
    """Build scale layout matching aiter/CK preshuffle for MXFP4 MFMA kernels.
    scale dtype is e8m0
    the scale shuffle to [K_Pack, N_Pack], pack to int32

    Shape: (N1, K1, KLane, NLane, [K_Pack, N_Pack]) = (N/32, K/8, 4, 16, [2, 2])
    """
    c16 = arith.constant(16, index=True)
    c32 = arith.constant(32, index=True)
    c4 = arith.constant(4, index=True)

    c_mn_pack = arith.constant(mn_pack, index=True)
    c_k_pack = arith.constant(k_pack, index=True)
    c_k_scale = c_k / scale_block_size

    c_mn1 = c_mn / c16 / c_mn_pack
    c_k1 = c_k_scale / c4 / c_k_pack

    # We keep the same 64B K0 "macro-step" used by CK/aiter preshuffle.
    if elem_bytes != mn_pack * k_pack:
        raise ValueError(f"elem_bytes of scale must be {mn_pack} * {k_pack}, got {elem_bytes!r}")

    # Strides derived from the layout shape:
    # - KPack stride = 1
    # - NLane stride = KPackElems
    # - KLane stride = NLane * KPackElems = 16 * KPackElems
    # - K0   stride = KLane * NLane * KPackElems = 4 * 16 * KPackElems
    stride_nlane = arith.constant(1, index=True)
    stride_klane = c16
    stride_k0 = c4 * stride_klane
    stride_n0 = c_k1 * stride_k0

    stride_b_scale = (
        stride_n0,      # n0
        stride_k0,      # k0
        stride_klane,   # KLane
        stride_nlane,   # NLane
    )
    layout_b = flir.make_layout((c_mn1, c_k1, c4, c16), stride=stride_b_scale)
    return layout_b


def load_b_pack_k32(
    buffer_ops,
    flir,
    arith,
    vector,
    *,
    arg_b,
    b_rsrc,
    layout_b,
    base_k: ir.Value,
    ki_step: int,
    n_blk: ir.Value,
    n_intra: ir.Value,
    lane_div_16: ir.Value,
    elem_type: ir.Type,
    kpack_bytes: int = 16,
    elem_bytes: int = 1,
    unpack_int4: bool = False,
) -> ir.Value:
    """Load one B pack for one MFMA(x32) micro-step.

    Returns an i64 Value containing 8 bytes consumed by MFMA.

    - For FP8/INT8: loads 16 bytes (one full KPack) and extracts the 8 bytes used by
      this micro-step.
    - For packed INT4 (W4A8): loads 4 bytes (8 int4 values) and unpacks to 8 int8 bytes
      using the 7-op sequence (no v_perm).
    """
    if kpack_bytes not in (8, 16):
        raise ValueError(f"kpack_bytes must be 8 or 16, got {kpack_bytes!r}")
    if unpack_int4 and kpack_bytes != 8:
        raise ValueError("unpack_int4 requires kpack_bytes=8 (packed int4 layout)")

    if elem_bytes not in (1, 2):
        raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")

    c64 = arith.constant(64, index=True)
    base_k_bytes = base_k * arith.constant(int(elem_bytes), index=True)
    k0_base = base_k_bytes / c64
    k0 = k0_base + arith.constant(ki_step // 2, index=True)
    k1 = lane_div_16
    half_bytes = kpack_bytes // 2
    k2_base = arith.constant((ki_step % 2) * half_bytes, index=True)

    # Always compute the *pack base* index (k2=0). Layout is in ELEMENT units.
    # add/sub on the address path and keeps the load address stable across the
    # two half-steps.
    coord_pack = flir.make_coord(n_blk, k0, k1, n_intra, arith.constant(0, index=True))
    idx_pack = flir.crd2idx(coord_pack, layout_b)

    if unpack_int4:
        # Load 4 bytes -> i32 -> unpack to i64 (8 i8 bytes).
        atom = flir.make_copy_atom(elem_type, vector_size=4)
        # packed int4 is byte-addressed (elem_bytes==1)
        idx_bytes = idx_pack + k2_base
        b_view = flir.TensorView(
            arg_b,
            (4,),
            strides=(1,),
            base_indices=(idx_bytes,),
            element_type=elem_type,
        )
        b4 = flir.copy(
            atom,
            b_view,
            None,
            alignment=4,
            return_vector=True,
            src_buffer_resource=b_rsrc,
            src_buffer_offset_in_bytes=True,
        )
        vec1_i32 = ir.VectorType.get([1], ir.IntegerType.get_signless(32))
        packed32 = vector.extract(
            vector.bitcast(vec1_i32, b4),
            static_position=[0],
            dynamic_position=[],
        )

        # 7-op unpack (and + mul + and_or + shifts). Requires prepacked nibble layout:
        # bytes: [ (v4<<4)|v0, (v5<<4)|v1, (v6<<4)|v2, (v7<<4)|v3 ]
        c_08080808 = arith.constant(0x08080808, type=ir.IntegerType.get_signless(32))
        c_0f0f0f0f = arith.constant(0x0F0F0F0F, type=ir.IntegerType.get_signless(32))
        c_1e = arith.constant(0x1E, type=ir.IntegerType.get_signless(32))
        c_4_i32 = arith.constant(4, type=ir.IntegerType.get_signless(32))

        s0 = (packed32 & c_08080808) * c_1e
        even = (packed32 & c_0f0f0f0f) | s0

        t = packed32 >> c_4_i32
        s1 = (t & c_08080808) * c_1e
        odd = (t & c_0f0f0f0f) | s1

        vec2_i32 = ir.VectorType.get([2], ir.IntegerType.get_signless(32))
        v2 = vector.from_elements(vec2_i32, [even, odd])
        vec1_i64 = ir.VectorType.get([1], ir.IntegerType.get_signless(64))
        v64 = vector.bitcast(vec1_i64, v2)
        return vector.extract(v64, static_position=[0], dynamic_position=[])

    # FP8/INT8: load 16 bytes (one full KPack) and extract half (8B) as i64.
    #
    # This keeps the original semantics (return the same 8B i64 used by MFMA for
    # this `ki_step`), but makes the intended 16B buffer-load (dwordx4) explicit
    # in the IR instead of relying on backend vectorization.
    vec_elems = kpack_bytes // int(elem_bytes)
    atom = flir.make_copy_atom(elem_type, vector_size=vec_elems)
    b_view = flir.TensorView(
        arg_b,
        (vec_elems,),
        strides=(1,),
        base_indices=(idx_pack,),
        element_type=elem_type,
    )
    b16 = flir.copy(
        atom,
        b_view,
        None,
        # Keep conservative alignment here: some layouts/launchers may only guarantee 8B.
        # This is still compatible with 16B buffer loads; it just avoids overstating
        # alignment to the compiler.
        alignment=8,
        return_vector=True,
        src_buffer_resource=b_rsrc,
        # Only 1B element types can safely treat the base index as bytes.
        src_buffer_offset_in_bytes=(elem_bytes == 1),
    )
    # Extract the needed 8B half as an i64 while keeping the other half dead.
    #
    # NOTE: We intentionally build the i64 from the selected 2 dwords, instead of
    # `bitcast -> i64x2 -> extract`, to help the backend shorten live ranges and
    # avoid unnecessary VGPR pressure on some schedules.
    i32 = ir.IntegerType.get_signless(32)
    i64 = ir.IntegerType.get_signless(64)
    vec4_i32 = ir.VectorType.get([4], i32)
    b_i32x4 = vector.bitcast(vec4_i32, b16)

    half = ki_step % 2
    if half == 0:
        d0 = vector.extract(b_i32x4, static_position=[0], dynamic_position=[])
        d1 = vector.extract(b_i32x4, static_position=[1], dynamic_position=[])
    else:
        d0 = vector.extract(b_i32x4, static_position=[2], dynamic_position=[])
        d1 = vector.extract(b_i32x4, static_position=[3], dynamic_position=[])

    vec2_i32 = ir.VectorType.get([2], i32)
    v2 = vector.from_elements(vec2_i32, [d0, d1])
    vec1_i64 = ir.VectorType.get([1], i64)
    v64 = vector.bitcast(vec1_i64, v2)
    return vector.extract(v64, static_position=[0], dynamic_position=[])


def tile_chunk_coord_i32(
    flir,
    arith,
    *,
    tx_i32_base: ir.Value,
    i: int,
    total_threads: int,
    layout_tile_div4,
    chunk_i32: int = 4,
):
    """Map (thread, chunk_id) -> (row_local, col_local_i32) for X/A loads.

    General form (dword-granularity):
      chunk_linear   = tx + i*total_threads
      chunk_i32_base = chunk_linear * chunk_i32

    Where chunk_i32 is the number of dwords per chunk:
      - 4  -> 16B (dwordx4)
      - 2  ->  8B (dwordx2)
      - 1  ->  4B (dword)

    NOTE: `layout_tile_div4` is expressed in dword elements along K (K/4),
    matching the existing GEMM/MoE mapping.
    """
    if chunk_i32 not in (1, 2, 4):
        raise ValueError(f"chunk_i32 must be one of (1,2,4), got {chunk_i32!r}")
    chunk_off_i32 = arith.constant(i * total_threads * chunk_i32, index=True)
    tile_idx_i32 = tx_i32_base + chunk_off_i32
    coord_local = flir.idx2crd(tile_idx_i32, layout_tile_div4)
    row_local = flir.get(coord_local, 0)
    col_local_i32 = flir.get(coord_local, 1)
    return row_local, col_local_i32


def buffer_copy_gmem16_dwordx4(
    flir,
    *,
    arg,
    elem_type,
    idx_i32: ir.Value,
    atom_g2r16,
    rsrc,
    vec_elems: int = 16,
):
    """Copy 16 bytes from global memory into regs via buffer-load dwordx4 lowering.

    `idx_i32` is a dword element offset (not bytes), so `src_buffer_offset_in_bytes=False`.
    """
    if int(vec_elems) <= 0:
        raise ValueError(f"vec_elems must be > 0, got {vec_elems!r}")
    view = flir.TensorView(
        arg,
        (int(vec_elems),),
        strides=(1,),
        base_indices=(idx_i32,),
        element_type=elem_type,
    )
    return flir.copy(
        atom_g2r16,
        view,
        None,
        alignment=16,
        return_vector=True,
        src_buffer_resource=rsrc,
        src_buffer_offset_in_bytes=False,
    )


def lds_store_16b_xor16(
    flir,
    arith,
    vector,
    *,
    lds_memref,
    vec16_ty,
    elem_type,
    atom_s16,
    layout_lds,
    row_local: ir.Value,
    col_local_i32: ir.Value,
    tx_c4: ir.Value,
    k_blocks16: ir.Value,
    lds_base: ir.Value,
    vec_part_i32x4: ir.Value,
    elem_bytes: int = 1,
):
    """Store one 16B chunk into LDS with CK-style XOR16 swizzle on the K dimension."""
    if elem_bytes not in (1, 2):
        raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")
    col_local_bytes = col_local_i32 * tx_c4
    col_swz_bytes = flir.swizzle_xor16(row_local, col_local_bytes, k_blocks16)
    col_swz = col_swz_bytes if elem_bytes == 1 else (col_swz_bytes / arith.constant(2, index=True))
    coord_store = flir.make_coord(row_local, col_swz)
    idx0 = flir.crd2idx(coord_store, layout_lds)
    idx0 = idx0 + lds_base
    v16 = vector.bitcast(vec16_ty, vec_part_i32x4)
    extent_elems = 16 if elem_bytes == 1 else 8
    s_view = flir.TensorView(
        lds_memref,
        (extent_elems,),
        strides=(1,),
        base_indices=(idx0,),
        element_type=elem_type,
    )
    flir.copy(atom_s16, v16, s_view, alignment=16)


def lds_store_8b_xor16(
    flir,
    arith,
    vector,
    *,
    lds_memref,
    vec8_ty,
    elem_type,
    atom_s8,
    layout_lds,
    row_local: ir.Value,
    col_local_i32: ir.Value,
    tx_c4: ir.Value,
    k_blocks16: ir.Value,
    lds_base: ir.Value,
    vec_part_i32x2: ir.Value,
    elem_bytes: int = 1,
):
    """Store one 8B chunk into LDS with CK-style XOR16 swizzle on the K dimension."""
    if elem_bytes not in (1, 2):
        raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")
    col_local_bytes = col_local_i32 * tx_c4
    col_swz_bytes = flir.swizzle_xor16(row_local, col_local_bytes, k_blocks16)
    col_swz = col_swz_bytes if elem_bytes == 1 else (col_swz_bytes / arith.constant(2, index=True))
    coord_store = flir.make_coord(row_local, col_swz)
    idx0 = flir.crd2idx(coord_store, layout_lds)
    idx0 = idx0 + lds_base
    v8 = vector.bitcast(vec8_ty, vec_part_i32x2)
    extent_elems = 8 if elem_bytes == 1 else 4
    s_view = flir.TensorView(
        lds_memref,
        (extent_elems,),
        strides=(1,),
        base_indices=(idx0,),
        element_type=elem_type,
    )
    flir.copy(atom_s8, v8, s_view, alignment=8)


def lds_store_4b_xor16(
    flir,
    arith,
    vector,
    *,
    lds_memref,
    vec4_ty,
    elem_type,
    atom_s4,
    layout_lds,
    row_local: ir.Value,
    col_local_i32: ir.Value,
    tx_c4: ir.Value,
    k_blocks16: ir.Value,
    lds_base: ir.Value,
    vec_part_i32x1: ir.Value,
    elem_bytes: int = 1,
):
    """Store one 4B chunk into LDS with CK-style XOR16 swizzle on the K dimension."""
    if elem_bytes not in (1, 2):
        raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")
    col_local_bytes = col_local_i32 * tx_c4
    col_swz_bytes = flir.swizzle_xor16(row_local, col_local_bytes, k_blocks16)
    col_swz = col_swz_bytes if elem_bytes == 1 else (col_swz_bytes / arith.constant(2, index=True))
    coord_store = flir.make_coord(row_local, col_swz)
    idx0 = flir.crd2idx(coord_store, layout_lds)
    idx0 = idx0 + lds_base
    v4 = vector.bitcast(vec4_ty, vec_part_i32x1)
    extent_elems = 4 if elem_bytes == 1 else 2
    s_view = flir.TensorView(
        lds_memref,
        (extent_elems,),
        strides=(1,),
        base_indices=(idx0,),
        element_type=elem_type,
    )
    flir.copy(atom_s4, v4, s_view, alignment=4)


def lds_load_pack_k32(
    flir,
    arith,
    vector,
    *,
    lds_memref,
    layout_lds,
    k_blocks16: ir.Value,
    curr_row_a_lds: ir.Value,
    col_base: ir.Value,
    half: int,
    lds_base: ir.Value,
    ck_lds128: bool,
    vec16_ty,
    vec8_ty,
    vec2_i64_ty,
    vec1_i64_ty,
):
    """Load one i64 A-pack for an MFMA K32 micro-step from LDS.

    - ck_lds128=True: load 16B and extract half (8B) as i64
    - ck_lds128=False: load 8B directly as i64
    """
    col_base_swz = flir.swizzle_xor16(curr_row_a_lds, col_base, k_blocks16)
    if ck_lds128:
        coord_a16 = flir.make_coord(curr_row_a_lds, col_base_swz)
        idx_a16 = flir.crd2idx(coord_a16, layout_lds)
        idx_a16 = idx_a16 + lds_base
        loaded_a16 = vector.load_op(vec16_ty, lds_memref, [idx_a16])
        a_vec128 = vector.bitcast(vec2_i64_ty, loaded_a16)
        return vector.extract(a_vec128, static_position=[half], dynamic_position=[])
    else:
        col_swizzled = col_base_swz + arith.constant(int(half) * 8, index=True)
        coord_a = flir.make_coord(curr_row_a_lds, col_swizzled)
        idx_a = flir.crd2idx(coord_a, layout_lds)
        idx_a = idx_a + lds_base
        loaded_a8 = vector.load_op(vec8_ty, lds_memref, [idx_a])
        a_vec64 = vector.bitcast(vec1_i64_ty, loaded_a8)
        return vector.extract(a_vec64, static_position=[0], dynamic_position=[])


__all__ = [
    "PreshuffleBLayout",
    "buffer_copy_gmem16_dwordx4",
    "lds_load_pack_k32",
    "lds_store_4b_xor16",
    "lds_store_8b_xor16",
    "lds_store_16b_xor16",
    "make_preshuffle_b_layout",
    "make_preshuffle_scale_layout",
    "load_b_pack_k32",
    "tile_chunk_coord_i32",
]

