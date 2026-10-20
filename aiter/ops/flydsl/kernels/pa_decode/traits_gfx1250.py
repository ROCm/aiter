# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 FlyDSL Project Contributors


"""Wave32 tile policy, dtypes and LDS sizing for gfx1250 PA decode."""

from dataclasses import dataclass

import flydsl.expr as fx


@dataclass(frozen=True)
class Gfx1250Traits:
    head_dim: int
    query_group_size: int
    block_size: int
    softmax_scale: float
    query_dtype: str
    per_token_kv: bool
    query_length: int
    trans_v: bool
    sliding_window: int
    compute_dtype: str = "fp8"
    direct_output: bool = False
    query_splits: int = 1

    WAVE = 32
    NWARP = 4
    FP8_MAX = 448.0

    @property
    def TOKENS(self):
        # Page-sized transfers amortize page/scale staging and softmax exchange.
        # Windowed rows retain 64-token normalization for FP8 accuracy.
        return (
            128
            if self.packed_lds
            and self.head_dim in (128, 256)
            and self.block_size >= 128
            and self.sliding_window == 0
            else 64
        )

    @property
    def bf16(self):
        return self.compute_dtype == "bf16"

    @property
    def elem_bytes(self):
        return 2 if self.bf16 else 1

    @property
    def chunk(self):
        return 16 // self.elem_bytes

    @property
    def kv_dtype(self):
        return fx.BFloat16 if self.bf16 else fx.Uint8

    @property
    def rows(self):
        return self.query_length * self.query_group_size // self.query_splits

    @property
    def m_tiles(self):
        return (self.rows + 15) // 16

    @property
    def q_stride(self):
        return self.head_dim * self.elem_bytes + 16

    @property
    def q_tiles(self):
        # The widest BF16 MTP tile fits by staging one query tile at a time.
        full = self.m_tiles * 16 * self.q_stride
        return (
            1 if full + self.kv_stride + self.p_bytes + 576 > 327680 else self.m_tiles
        )

    @property
    def q_bytes(self):
        return self.q_tiles * 16 * self.q_stride

    @property
    def packed_lds(self):
        # FP8 cache layout is independent of scalar or per-token scales.
        # Two fused query tiles reuse the packed page and K128 WMMA path for
        # measured D256/G8/MTP4 full attention, without extra KV fetches.
        return (
            not self.bf16
            and self.head_dim in (128, 256)
            and self.block_size >= 64
            and self.trans_v
            and (
                self.m_tiles == 1
                or (
                    self.head_dim == 256
                    and self.block_size == 128
                    and self.query_length == 4
                    and self.query_group_size == 8
                    and self.sliding_window == 0
                )
            )
        )

    @property
    def k_bytes(self):
        return self.TOKENS * (self.head_dim if self.packed_lds else self.q_stride)

    @property
    def v_stride(self):
        return self.TOKENS * self.elem_bytes + 16

    @property
    def v_bytes(self):
        return self.head_dim * (self.TOKENS if self.packed_lds else self.v_stride)

    @property
    def kv_stride(self):
        return self.k_bytes + self.v_bytes + 2 * self.TOKENS * 4

    @property
    def p_bytes(self):
        return 16 * self.v_stride

    @property
    def async_page(self):
        # Two packed buffers overlap page DMA with QK/softmax/PV. Keep this
        # measured specialization separate from windowed and smaller-D tiles.
        return (
            self.packed_lds
            and self.head_dim == 256
            and self.block_size == 128
            and self.sliding_window == 0
        )

    @property
    def buffers(self):
        if self.packed_lds:
            return 2 if self.async_page else 1
        return (
            2 if self.q_bytes + 2 * self.kv_stride + self.p_bytes + 576 <= 327680 else 1
        )

    @property
    def p_offset(self):
        return self.q_bytes + self.buffers * self.kv_stride

    @property
    def stats_offset(self):
        return self.p_offset + self.p_bytes

    @property
    def total_bytes(self):
        return self.stats_offset + 576

    @property
    def use_tdm(self):
        # Packed FP8 tiles retain the cache layout for TDM transfers. Other
        # per-token D128+ tiles also benefit from scalar address generation.
        # BF16 descriptors fit D64–512; row padding requires a power-of-two
        # interval <= 1024 bytes.
        return (
            (
                self.bf16
                or self.packed_lds
                or (self.per_token_kv and self.head_dim >= 128)
            )
            and self.head_dim * self.elem_bytes <= 1024
            and self.head_dim & (self.head_dim - 1) == 0
        )

    @property
    def q_dtype(self):
        return fx.BFloat16 if self.query_dtype == "bf16" else fx.Float16
