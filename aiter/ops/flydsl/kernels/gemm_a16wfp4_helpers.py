# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Components for the gfx950 dense BF16 x MXFP4 GEMM."""

from dataclasses import dataclass

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .moe_2stage_a16wmix.utils import make_a_loader, make_b_loader
from .mxfp4_gemm_common import lds_typed_ptr

_NUM_WAVES = 4


@dataclass(frozen=True)
class DenseGemmTraits:
    n: int
    k: int
    block_m: int
    tile_n: int
    tile_k: int
    b_cache_mod: int
    k_wave: int = 1
    waves_per_eu: int | None = None

    def __post_init__(self):
        if self.k_wave not in (1, 2, 4):
            raise ValueError(f"k_wave must be 1, 2, or 4, got {self.k_wave}")
        if _NUM_WAVES % self.k_wave:
            raise ValueError(f"4 waves cannot be split by k_wave={self.k_wave}")
        if self.k % (self.k_wave * self.tile_k):
            raise ValueError(
                f"K={self.k} must be divisible by k_wave*TILE_K="
                f"{self.k_wave * self.tile_k}"
            )

    @property
    def a_tile_bytes(self):
        return self.tile_k * 2

    @property
    def num_n_waves(self):
        return _NUM_WAVES // self.k_wave

    @property
    def k_len(self):
        return self.k // self.k_wave

    @property
    def k_tiles(self):
        return self.k_len // self.tile_k

    @property
    def a_lds_stages(self):
        return 2 if self.k_tiles > 1 else 1

    @property
    def a_slot_bytes(self):
        return self.block_m * self.a_tile_bytes

    @property
    def m_repeats(self):
        return self.block_m // 16

    @property
    def k_repeats(self):
        return self.a_tile_bytes // 64

    @property
    def n_per_wave(self):
        return self.tile_n // self.num_n_waves

    @property
    def n_repeats(self):
        return self.n_per_wave // 16

    @property
    def n_blocks(self):
        return self.n // self.tile_n

    @property
    def a_lds_bytes(self):
        return self.k_wave * self.a_lds_stages * self.a_slot_bytes

    @property
    def k_reduce_bytes(self):
        if self.k_wave == 1:
            return 0
        nm = self.n_repeats * self.m_repeats
        return _NUM_WAVES * 64 * nm * 4 * 4

    @property
    def lds_bytes(self):
        return max(self.a_lds_bytes, self.k_reduce_bytes)

    @property
    def kernel_name(self):
        wpe = 0 if self.waves_per_eu is None else self.waves_per_eu
        return (
            f"dense_a16wfp4_bf16_gfx950_m{self.block_m}_n{self.n}_k{self.k}"
            f"_tn{self.tile_n}_tk{self.tile_k}_kw{self.k_wave}_wpe{wpe}"
        )

    @property
    def cache_key(self):
        return (
            self.n,
            self.k,
            self.block_m,
            self.tile_n,
            self.tile_k,
            self.b_cache_mod,
            self.k_wave,
            self.waves_per_eu,
        )


def make_dense_operand_loaders(
    traits,
    lds,
    arg_a,
    arg_bq,
    arg_bscale,
    m,
    m_row,
    n_row,
    lane,
    wave,
):
    lane_div_16 = lane // 16
    lane_mod_16 = lane % 16
    num_n_waves = traits.num_n_waves
    if const_expr(traits.k_wave > 1):
        wave_n_id = wave % fx.Int32(num_n_waves)
        wave_k_id = rocdl.readfirstlane(T.i32, wave // fx.Int32(num_n_waves))
        k_grp_base_bytes = wave_k_id * fx.Int32(
            traits.a_lds_stages * traits.a_slot_bytes
        )
    else:
        wave_n_id = wave
        wave_k_id = fx.Int32(0)
        k_grp_base_bytes = fx.Int32(0)

    b_loader = make_b_loader(
        arg_bq,
        arg_bscale,
        N_OUT=traits.n,
        K=traits.k,
        NE=1,
        e=fx.Int32(0),
        lane_div_16=lane_div_16,
        lane_mod_16=lane_mod_16,
        TILE_K=traits.tile_k,
        w_dtype="fp4",
        b_cache_mod=traits.b_cache_mod,
        use_k16=False,
    )
    a_loader = make_a_loader(
        lds,
        num_i32=traits.lds_bytes // 4,
        BM=traits.block_m,
        TILE_K=traits.tile_k,
        KH_TILE_BYTES=traits.a_tile_bytes,
        k_blocks16=traits.a_tile_bytes // 16,
        lane_div_16=lane_div_16,
        lane_mod_16=lane_mod_16,
        swizzle=True,
        a_ptr=arg_a,
        a_num_bytes=fx.Int64(m) * fx.Int64(traits.k * 2),
        a_load_threads=num_n_waves * 64,
        row_base_dwords=lambda row: (m_row + row) * fx.Int32(traits.k // 2),
        dma_cache_mod=0,
        dma_via_vgpr=False,
        k_grp_base_bytes=k_grp_base_bytes,
        A_SLOT_BYTES=traits.a_slot_bytes,
    )
    columns = [
        b_loader.col(
            n_row,
            wave_n_id * traits.n_per_wave,
            ni * 16,
        )
        for ni in range_constexpr(traits.n_repeats)
    ]
    return a_loader, b_loader, columns, wave_n_id, wave_k_id


def make_dense_accumulators(traits):
    layout = fx.make_layout(4, 1)
    accumulators = [
        [fx.make_rmem_tensor(layout, fx.Float32) for _ in range(traits.n_repeats)]
        for _ in range(traits.m_repeats)
    ]
    zero = Vec.filled(4, 0.0, fx.Float32)
    for mi in range_constexpr(traits.m_repeats):
        for ni in range_constexpr(traits.n_repeats):
            accumulators[mi][ni].store(zero)
    return accumulators


def load_b_tile(traits, b_loader, columns, base_k):
    """Packed weight + E8M0 scales for one K tile."""
    if const_expr(traits.block_m >= 64):
        # Issue the small scale loads first so they age while packed-weight VMEM runs.
        b_scale = [b_loader.load_scale(base_k, col) for col in columns]
        b_raw = [b_loader.load_raw(base_k, col) for col in columns]
    else:
        b_raw = [b_loader.load_raw(base_k, col) for col in columns]
        b_scale = [b_loader.load_scale(base_k, col) for col in columns]
    return b_raw, b_scale


def advance_k_tile(traits, a_loader, b_loader, columns, k_base, kt):
    """Stage the next A tile in LDS and issue the next B tile."""
    next_k = k_base + fx.Int32((kt + 1) * traits.tile_k)
    a_loader.store_tile(next_k, slot=(kt + 1) % traits.a_lds_stages)
    return load_b_tile(traits, b_loader, columns, next_k)


def load_a_frags(traits, a_loader, slot):
    """Read the staged A tile out of LDS into MFMA fragments."""
    return [
        [a_loader.load(mi, ku, slot=slot) for ku in range_constexpr(traits.k_repeats)]
        for mi in range_constexpr(traits.m_repeats)
    ]


def _mma_order(traits):
    """(n, k) visit order. Wide tiles cross N before advancing K, which stretches
    each accumulator's MFMA dependency distance from m_repeats to
    m_repeats * n_repeats."""
    if traits.block_m >= 64:
        return [
            (ni, ku) for ku in range(traits.k_repeats) for ni in range(traits.n_repeats)
        ]
    return [
        (ni, ku) for ni in range(traits.n_repeats) for ku in range(traits.k_repeats)
    ]


def compute_dense_tile(traits, mma, accumulators, b_loader, b_tile, a_frags):
    """Decode one B tile to BF16 and accumulate it into the FP32 tile."""
    b_raw, b_scale = b_tile
    a_fragment = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
    b_fragment = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
    for ni, ku in _mma_order(traits):
        b_fragment.store(b_loader.upconvert(b_raw[ni], ku, b_scale[ni][ku]))
        for mi in range_constexpr(traits.m_repeats):
            a_fragment.store(a_frags[mi][ku])
            fx.gemm(
                mma, accumulators[mi][ni], a_fragment, b_fragment, accumulators[mi][ni]
            )


def reduce_dense_k_wave(traits, accumulators, lds, wave, wave_n_id, lane):
    """Sum intra-block K-slice partials through the A-LDS region, now free."""
    nm = traits.n_repeats * traits.m_repeats
    grp_stride = 64 * nm * 4
    lds_base = fx.Int32(fx.ptrtoint(lds))
    scratch = lds_typed_ptr(lds_base, T.f32, align=4)
    peers = lds_typed_ptr(lds_base, T.f32, align=16)
    gpu.barrier()
    my_base = wave * fx.Int32(grp_stride) + lane * fx.Int32(4)
    for ai in range_constexpr(nm):
        acc = accumulators[ai // traits.n_repeats][ai % traits.n_repeats]
        v = Vec(fx.memref_load_vec(acc))
        sidx = my_base + fx.Int32(ai * 64 * 4)
        for vv in range_constexpr(4):
            fx.ptr_store(v[vv], scratch + (sidx + fx.Int32(vv)))
    gpu.barrier()
    for ai in range_constexpr(nm):
        acc = accumulators[ai // traits.n_repeats][ai % traits.n_repeats]
        s = Vec(fx.memref_load_vec(acc))
        ai_off = fx.Int32(ai * 64 * 4) + lane * fx.Int32(4)
        for g in range_constexpr(1, traits.k_wave):
            peer = fx.Int32(g * traits.num_n_waves) + wave_n_id
            pv = Vec(
                fx.ptr_load(
                    peers + (peer * fx.Int32(grp_stride) + ai_off),
                    result_type=T.vec(4, T.f32),
                )
            )
            s = Vec.from_elements(
                [s[vv] + pv[vv] for vv in range_constexpr(4)], fx.Float32
            )
        acc.store(s)


def make_dense_store(arg_out, n):
    """Buffer-copy state for the BF16 epilogue: (output, atom, fragment).

    The guarded store itself stays in the kernel body, where the flyc rewriter
    can turn its runtime row predicate into an scf.if.
    """
    pointer = fx.PointerType.get(
        fx.BFloat16.ir_type,
        address_space=fx.AddressSpace.Global,
        alignment=2,
    )
    view = fx.make_view(
        fx.inttoptr(pointer, arg_out),
        fx.make_layout((1 << 30, 1), (1, 1)),
    )
    output = fx.logical_divide(
        fx.rocdl.make_buffer_tensor(view, max_size=True),
        fx.make_layout(1, 1),
    )
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
    fragment = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.BFloat16)
    return output, atom, fragment


def store_dense_bf16(store, value, index):
    output, atom, fragment = store
    fragment.store(Vec.filled(1, value.to(fx.BFloat16), fx.BFloat16))
    fx.copy(atom, fragment, fx.slice(output, (None, fx.Int32(index))))
