# SPDX-License-Identifier: MIT
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Compile-time tile and layout traits.

Part of the gfx950 dual-wave fp8 (e4m3fn) flash-attention kernel, migrated from
FlyDSL ``kernels/attention/flash_attn_utils.py`` and restricted to the symbols
the fp8 path reaches. The bf16/f16 dual-wave, the gfx942 generic path, paged KV,
and the bias/ALiBi helpers are not part of it and were left behind.
"""

from dataclasses import dataclass

from aiter.ops.flydsl.kernels.fmha_gfx950.flash_attn_utils_primitives import (
    LDS_BYTES_GFX950,
)


@dataclass(frozen=True)
class DualwaveSwpFp8Traits:
    """Pure compile-time tile/layout constants for the gfx950 DUALWAVE_SWP fp8 kernel."""

    BLOCK_M: int
    BLOCK_N: int
    WARP_SIZE: int
    NUM_WAVES: int
    BLOCK_SIZE: int
    ROWS_PER_WAVE: int
    HEAD_DIM: int
    HEAD_DIM_V: int
    D_CHUNK: int
    D_CHUNKS: int
    PV_K_STEPS: int
    NUM_HEADS_Q: int
    NUM_HEADS_KV: int
    GQA_GROUP_SIZE: int
    CAUSAL: bool
    DTYPE_STR: str
    WAVES_PER_EU: int
    DAZ: bool
    DUALWAVE_SWP_LAZY_RESCALE: bool
    DUALWAVE_SWP_SETPRIO: bool
    DUALWAVE_SWP_ENABLE_STAGGER: bool
    NUM_KV_SPLITS: int
    SPLITK: bool
    VARLEN: bool
    CROSS_SEQLEN: bool
    DEFAULT_STRIDE_Q_N: int
    DEFAULT_STRIDE_KV_N: int
    DEFAULT_STRIDE_V_N: int
    DEFAULT_STRIDE_O_N: int
    QLDS: bool
    K_BAND_CHUNK: tuple[int, ...]
    K_BAND_BASE: tuple[int, ...]
    K_BAND_LINE_STRIDE: tuple[int, ...]
    K_BAND_GLOBAL_D: tuple[int, ...]
    K_WS_BAND: tuple[int, ...]
    K_WS_OFF: tuple[int, ...]
    DMA_BYTES: int
    ELEM_BYTES: int
    OUT_ELEM_BYTES: int
    VEC_KV: int
    LANE_SPLIT_KV: int
    SMEM_K_TILE_ELEMS: int
    NUM_PREFETCH_K: int
    DUALWAVE_SWP_KV_PER_BUFFER: int
    LDS_KV_TOTAL_SIZE: int
    DUALWAVE_SWP_K_BUF_BASE: tuple[int, int]
    DUALWAVE_SWP_V_BUF_BASE: tuple[int, int]
    VT_BF16_TOTAL: int
    DUALWAVE_SWP_RESCALE_THRESHOLD: float
    SCHED_MFMA_MASK: int
    SCHED_DS_READ_MASK: int
    NEG_INF_F32_BITS: int
    XCD_SWIZZLE: bool = False
    BATCH_INTERLEAVE_GROUP: int = 1

    @property
    def cache_tag(self):
        return (
            self.NUM_HEADS_Q,
            self.NUM_HEADS_KV,
            self.HEAD_DIM,
            self.CAUSAL,
            self.DTYPE_STR,
            self.WAVES_PER_EU,
            self.DAZ,
            self.DUALWAVE_SWP_LAZY_RESCALE,
            self.DUALWAVE_SWP_RESCALE_THRESHOLD,
            self.DUALWAVE_SWP_SETPRIO,
            self.DUALWAVE_SWP_ENABLE_STAGGER,
            self.NUM_KV_SPLITS,
            self.SPLITK,
            self.VARLEN,
            self.CROSS_SEQLEN,
            self.HEAD_DIM_V,
            self.QLDS,
            self.K_BAND_CHUNK,
            "fp8_wide_qk_hiprec_pv",
            self.ELEM_BYTES,
            self.OUT_ELEM_BYTES,
            self.LANE_SPLIT_KV,
            self.VT_BF16_TOTAL,
            self.NUM_PREFETCH_K,
            self.XCD_SWIZZLE,
            self.BATCH_INTERLEAVE_GROUP,
            self.BLOCK_M,
            self.BLOCK_SIZE,
            self.NUM_WAVES,
        )


def _make_dualwave_swp_fp8_traits(
    num_heads,
    num_kv_heads,
    head_dim,
    rescale_threshold,
    head_dim_v=None,
    block_m=256,
    causal=True,
    waves_per_eu=2,
    daz=True,
    dualwave_swp_lazy_rescale=True,
    dualwave_swp_setprio=True,
    dualwave_swp_enable_stagger=True,
    num_kv_splits=1,
    varlen=False,
    cross_seqlen=False,
    xcd_swizzle=False,
    batch_interleave_group=1,
):
    """Build gfx950 DUALWAVE_SWP fp8 compile-time layout traits.

    ``head_dim`` is the QK reduction width (a multiple of 64: the QK MFMA is
    32x32x64) and ``head_dim_v`` the V/output width, tiled in 32-wide D_CHUNKs.
    """
    if head_dim_v is None:
        head_dim_v = head_dim
    if head_dim % 64:
        raise RuntimeError(
            f"fp8 flash attention needs head_dim % 64 == 0, got head_dim={head_dim}"
        )
    # D_CHUNKS == head_dim_v // 32 must land in [2, 6]: below 2 `_anchor_v_o`
    # aborts LLVM, above 6 the high D_CHUNKs come back wrong.
    if head_dim_v % 32 or not 64 <= head_dim_v <= 192:
        raise RuntimeError(
            "fp8 flash attention needs 64 <= head_dim_v <= 192 and head_dim_v % 32 == 0, "
            f"got head_dim_v={head_dim_v} (head_dim={head_dim})"
        )
    block_n = 64
    k_sub_n = 32
    warp_size = 64
    rows_per_wave = 32
    if block_m % rows_per_wave or block_m // rows_per_wave not in (4, 8):
        raise RuntimeError(
            f"fp8 flash attention supports block_m 128 (4 waves) or 256 (8 waves), got {block_m}"
        )
    num_waves = block_m // rows_per_wave
    block_size = num_waves * warp_size

    d_chunk = 32
    d_chunks = head_dim_v // d_chunk
    pv_k_step = 16
    pv_k_steps = k_sub_n // pv_k_step

    gqa_group_size = num_heads // num_kv_heads
    default_stride_q_n = num_heads * head_dim
    default_stride_kv_n = num_kv_heads * head_dim
    default_stride_v_n = num_kv_heads * head_dim_v
    default_stride_o_n = num_heads * head_dim_v

    # fp8: Q/K/V are 1B; O is bf16 (2B). ELEM_BYTES=1 drives the fp8 address math.
    elem_bytes = 1
    out_elem_bytes = 2
    vec_kv = 16 // elem_bytes
    lane_split_kv = 8
    smem_k_pad = 16 // elem_bytes

    rows_per_wave_dma = -(-block_n // num_waves)
    k_band_chunk, k_band_base, k_band_line_stride, k_band_global_d = [], [], [], []
    _off, _cursor = 0, 0
    while _off < head_dim:
        chunk = min(128, head_dim - _off)
        line_stride = rows_per_wave_dma * chunk + smem_k_pad
        k_band_chunk.append(chunk)
        k_band_base.append(_cursor)
        k_band_line_stride.append(line_stride)
        k_band_global_d.append(_off)
        _cursor += num_waves * line_stride
        _off += chunk
    smem_k_tile_elems = _cursor
    k_ws_band, k_ws_off = [], []
    for _bi, chunk in enumerate(k_band_chunk):
        for _o in range(0, chunk, 64):
            k_ws_band.append(_bi)
            k_ws_off.append(_o)
    num_prefetch_k = 6
    dualwave_swp_kv_per_buffer = smem_k_tile_elems
    lds_kv_total_size = num_prefetch_k * dualwave_swp_kv_per_buffer
    dualwave_swp_k_buf_base = tuple(
        i * dualwave_swp_kv_per_buffer for i in range(num_prefetch_k)
    )
    dualwave_swp_v_buf_base = tuple(
        smem_k_tile_elems + i * dualwave_swp_kv_per_buffer
        for i in range(num_prefetch_k)
    )

    # The +128 covers the alignment the DMA base is rounded up to.
    eb_bf = 2
    fp8_v_tile_bytes = (block_n // 8) * (head_dim_v // 16) * 128
    vt_bf16_total = num_prefetch_k * (fp8_v_tile_bytes // eb_bf) + 128

    splitk = num_kv_splits > 1

    qlds = head_dim <= 128

    lds_bytes = lds_kv_total_size * elem_bytes + vt_bf16_total * eb_bf
    if qlds:
        lds_bytes += block_m * head_dim * elem_bytes
    if lds_bytes > LDS_BYTES_GFX950:
        raise RuntimeError(
            f"fp8 flash attention head_dim={head_dim}/head_dim_v={head_dim_v} at block_m={block_m} "
            f"needs {lds_bytes} B of LDS, over the {LDS_BYTES_GFX950} B gfx950 workgroup limit. "
            "Largest head_dim_v that fits: 192 at head_dim 64/128/192, 160 at 256, 96 at 320; "
            "head_dim 384 and above never fits."
        )

    return DualwaveSwpFp8Traits(
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        WARP_SIZE=warp_size,
        NUM_WAVES=num_waves,
        BLOCK_SIZE=block_size,
        ROWS_PER_WAVE=rows_per_wave,
        HEAD_DIM=head_dim,
        HEAD_DIM_V=head_dim_v,
        D_CHUNK=d_chunk,
        D_CHUNKS=d_chunks,
        PV_K_STEPS=pv_k_steps,
        NUM_HEADS_Q=num_heads,
        NUM_HEADS_KV=num_kv_heads,
        GQA_GROUP_SIZE=gqa_group_size,
        CAUSAL=causal,
        DTYPE_STR="fp8",
        WAVES_PER_EU=waves_per_eu,
        DAZ=bool(daz),
        DUALWAVE_SWP_LAZY_RESCALE=bool(dualwave_swp_lazy_rescale),
        DUALWAVE_SWP_SETPRIO=bool(dualwave_swp_setprio),
        DUALWAVE_SWP_ENABLE_STAGGER=bool(dualwave_swp_enable_stagger),
        NUM_KV_SPLITS=num_kv_splits,
        SPLITK=splitk,
        VARLEN=bool(varlen),
        CROSS_SEQLEN=bool(cross_seqlen),
        DEFAULT_STRIDE_Q_N=default_stride_q_n,
        DEFAULT_STRIDE_KV_N=default_stride_kv_n,
        DEFAULT_STRIDE_V_N=default_stride_v_n,
        DEFAULT_STRIDE_O_N=default_stride_o_n,
        QLDS=bool(qlds),
        K_BAND_CHUNK=tuple(k_band_chunk),
        K_BAND_BASE=tuple(k_band_base),
        K_BAND_LINE_STRIDE=tuple(k_band_line_stride),
        K_BAND_GLOBAL_D=tuple(k_band_global_d),
        K_WS_BAND=tuple(k_ws_band),
        K_WS_OFF=tuple(k_ws_off),
        DMA_BYTES=16,
        ELEM_BYTES=elem_bytes,
        OUT_ELEM_BYTES=out_elem_bytes,
        VEC_KV=vec_kv,
        LANE_SPLIT_KV=lane_split_kv,
        SMEM_K_TILE_ELEMS=smem_k_tile_elems,
        NUM_PREFETCH_K=num_prefetch_k,
        DUALWAVE_SWP_KV_PER_BUFFER=dualwave_swp_kv_per_buffer,
        LDS_KV_TOTAL_SIZE=lds_kv_total_size,
        DUALWAVE_SWP_K_BUF_BASE=dualwave_swp_k_buf_base,
        DUALWAVE_SWP_V_BUF_BASE=dualwave_swp_v_buf_base,
        VT_BF16_TOTAL=vt_bf16_total,
        DUALWAVE_SWP_RESCALE_THRESHOLD=rescale_threshold,
        SCHED_MFMA_MASK=0x008,
        SCHED_DS_READ_MASK=0x100,
        NEG_INF_F32_BITS=0xFF800000,
        XCD_SWIZZLE=bool(xcd_swizzle),
        BATCH_INTERLEAVE_GROUP=int(batch_interleave_group),
    )


def dualwave_fp8_dma_per_iter(traits):
    rows_per_wave = -(-traits.BLOCK_N // traits.NUM_WAVES)
    k_instr = sum(
        -(-(rows_per_wave * (chunk // traits.VEC_KV)) // traits.WARP_SIZE)
        for chunk in traits.K_BAND_CHUNK
    )
    num_dma_v = (traits.BLOCK_N * (traits.HEAD_DIM_V // 16) * 16) // (
        traits.WARP_SIZE * traits.VEC_KV * traits.ELEM_BYTES
    )
    v_instr_min = num_dma_v // traits.NUM_WAVES
    return 2 * k_instr + 2 * v_instr_min
