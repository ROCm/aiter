# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""MXFP6-E2M3 packing utilities for the gfx950 MHA v4 kernels."""

import numpy as np

try:
    import torch
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except ImportError:
    _HAVE_TRITON = False


FP6_K_TILE_TOKENS = 128
FP6_K_PACKED_ROW_BYTES = 96
FP6_K_BUFFER_SLACK_BYTES = 256
FP6_K_SCALE_VALUES_PER_TOKEN = 4
FP6_K_SCALE_BUFFER_SLACK_BYTES = 64

_K_TILE_TOKENS = FP6_K_TILE_TOKENS
_K_PACKED_ROW_BYTES = FP6_K_PACKED_ROW_BYTES
_K_COMPACT_DATA_BYTES = _K_TILE_TOKENS * _K_PACKED_ROW_BYTES
_K_RESERVED_BYTES = 4096
_K_SCALE_TAIL_BYTES = 1024
_K_SCALE_TAIL_OFFSET = _K_COMPACT_DATA_BYTES + _K_RESERVED_BYTES
FP6_K_TILE_BYTES = _K_SCALE_TAIL_OFFSET + _K_SCALE_TAIL_BYTES
_K_TILE_BYTES = FP6_K_TILE_BYTES
_K_SEQ_STRIDE_BYTES = _K_TILE_BYTES // _K_TILE_TOKENS


def fp6_k_raw_buffer_sizes(batch, sequence, heads, tile=FP6_K_TILE_TOKENS):
    """Return contiguous data/scale buffer sizes for the gfx950 FP6 K ABI.

    Each head stores one 17,408-byte record per 128-token tile. The data buffer's
    256-byte tail covers the final shifted scale-tail read; the separate scale ABI
    buffer retains four E8M0 bytes per token plus 64 bytes of view slack.
    """
    tiles = (sequence + tile - 1) // tile
    data_size = batch * heads * tiles * FP6_K_TILE_BYTES + FP6_K_BUFFER_SLACK_BYTES
    scale_size = (
        batch * sequence * heads * FP6_K_SCALE_VALUES_PER_TOKEN
        + FP6_K_SCALE_BUFFER_SLACK_BYTES
    )
    return data_size, scale_size


def _k_lds_order_gather_index():
    """Per-tile [12288] token-major byte index for the compact LDS-order image."""
    c0 = np.arange(8192)
    block = np.arange(8)[:, None, None, None]
    parity = np.arange(2)[None, :, None, None]
    lane = np.arange(32)[None, None, :, None]
    byte = np.arange(8)[None, None, None, :]
    c1 = 8192 + block * 1024 + parity * 512 + lane * 16 + byte
    P = np.concatenate((c0, c1.reshape(-1)))
    byte = P & 15
    r = P >> 4
    v0 = r & 63
    r2 = r >> 6
    wv = r2 & 3
    iv = r2 >> 2
    v_k_base = (v0 & 31) * 96 + ((v0 >> 5) & 1) * 24
    blk = wv + (iv & 1) * 4
    half = blk & 1
    n = blk >> 1
    chunk = iv >> 1
    c_i = n * 32 * 96 + half * 48 + chunk * 16
    return (v_k_base + c_i + byte).astype(np.int64)


_TR8_SIGMA32 = np.array(
    [
        0,
        1,
        2,
        3,
        16,
        17,
        18,
        19,
        4,
        5,
        6,
        7,
        20,
        21,
        22,
        23,
        8,
        9,
        10,
        11,
        24,
        25,
        26,
        27,
        12,
        13,
        14,
        15,
        28,
        29,
        30,
        31,
    ],
    dtype=np.int64,
)


def _v_field_perm() -> np.ndarray:
    """Per-output-field source index into a 32-kv MX block.

    Combines (a) the cvt field interleave field[2i]=blk[i], field[2i+1]=blk[16+i]
    and (b) the tr8 within-block kv scramble, so loading the 32 values in this
    order yields the fp6 fields already in their final packed positions (groups of
    4 contiguous fields = 3 contiguous bytes, no further permutation)."""
    inv32 = np.empty(32, dtype=np.int64)
    inv32[_TR8_SIGMA32] = np.arange(32)
    c = np.where(np.arange(32) % 2 == 0, np.arange(32) // 2, 16 + np.arange(32) // 2)
    return inv32[c].astype(np.int32)  # fieldperm[f] = inv32[c(f)]


_V_KVTAB_CACHE: dict = {}


def _v_kvtab_dev(device, direct_p: bool):
    """Return the cached per-device field-to-KV permutation for V packing."""
    key = (device, direct_p)
    kvtab = _V_KVTAB_CACHE.get(key)
    if kvtab is None:
        table = _v_direct_kvtab() if direct_p else _v_noswap_kvtab()
        kvtab = torch.from_numpy(table.reshape(-1)).to(device)
        _V_KVTAB_CACHE[key] = kvtab
    return kvtab


def quantize_fp6_v_clean_triton(
    v_fp8: "torch.Tensor",
    tile: int = 128,
    direct_p: bool = False,
    fixed_e8m0: bool = False,
):
    """Pack FP8 V into combined FP6 data and E8M0 scale tiles."""
    assert _HAVE_TRITON, "triton/torch unavailable"
    b, sk, h_kv, d = v_fp8.shape
    assert d == 128 and tile == 128 and sk % tile == 0, (d, sk, tile)
    nT = sk // tile
    n_blocks = b * h_kv * nT * 128 * 4
    out = torch.empty(b * h_kv * nT * 12800, dtype=torch.uint8, device=v_fp8.device)
    kvtab = _v_kvtab_dev(v_fp8.device, direct_p)
    BLOCK_N = 128
    grid = (triton.cdiv(n_blocks, BLOCK_N),)
    _pack_v_fp6_kernel[grid](
        v_fp8,
        out,
        out,
        kvtab,
        v_fp8.stride(0),
        v_fp8.stride(1),
        v_fp8.stride(2),
        v_fp8.stride(3),
        sk,
        h_kv,
        nT,
        n_blocks,
        CLAMP_TAIL=False,
        FIXED_E8M0=fixed_e8m0,
        SEPARATE_OUTPUT=False,
        BLOCK_N=BLOCK_N,
    )
    return out.view(b, h_kv, nT * 12800)


def quantize_fp6_v_data_scale_triton(
    v_fp8: "torch.Tensor", tile: int = 128, fixed_e8m0: bool = False
):
    """Pack F8F6 V directly into its separate data and scale ABI buffers."""
    assert _HAVE_TRITON, "triton/torch unavailable"
    b, sk, h_kv, d = v_fp8.shape
    assert d == 128 and tile == 128, (d, sk, tile)
    nT = (sk + tile - 1) // tile
    n_blocks = b * h_kv * nT * 128 * 4
    data = torch.empty(
        b * h_kv * nT * 12288 + 256, dtype=torch.uint8, device=v_fp8.device
    )
    scale = torch.empty(b * h_kv * nT * 512, dtype=torch.uint8, device=v_fp8.device)
    kvtab = _v_kvtab_dev(v_fp8.device, True)
    BLOCK_N = 128
    grid = (triton.cdiv(n_blocks, BLOCK_N),)
    _pack_v_fp6_kernel[grid](
        v_fp8,
        data,
        scale,
        kvtab,
        v_fp8.stride(0),
        v_fp8.stride(1),
        v_fp8.stride(2),
        v_fp8.stride(3),
        sk,
        h_kv,
        nT,
        n_blocks,
        CLAMP_TAIL=sk % tile != 0,
        FIXED_E8M0=fixed_e8m0,
        SEPARATE_OUTPUT=True,
        BLOCK_N=BLOCK_N,
    )
    return data, scale


_NOSWAP_KVTAB_CACHE = None


def _v_noswap_kvtab() -> np.ndarray:
    """Return the field-to-KV map for the pre-swap P operand layout."""
    global _NOSWAP_KVTAB_CACHE
    if _NOSWAP_KVTAB_CACHE is not None:
        return _NOSWAP_KVTAB_CACHE
    fperm = _v_field_perm()
    srcL = np.zeros((64, 32), np.int64)
    srcF = np.zeros((64, 32), np.int64)
    for L in range(64):
        hi = L >= 32
        base = L - 32 if hi else L
        for f in range(32):
            even = (f % 2) == 0
            if not hi:
                srcL[L, f], srcF[L, f] = (L, f) if even else (L + 32, f - 1)
            else:
                srcL[L, f], srcF[L, f] = (base, f + 1) if even else (L, f)
    kvtab = 32 * (srcL // 32) + fperm[srcF]
    _NOSWAP_KVTAB_CACHE = kvtab.astype(np.int32)
    return _NOSWAP_KVTAB_CACHE


def _v_direct_kvtab() -> np.ndarray:
    """Scaled FP6-src0 field to logical KV for the direct FP8 P operand.

    A gfx950 one-hot probe shows FP6 physical contraction index ``a`` pairs with
    FP8 index ``swap_bits_4_5(a)`` for all 64 indices. The live P pack maps its
    physical byte ``s`` and lane group ``g`` to logical KV as
    ``32*(s//16) + 8*((s%16)//4) + s%4 + 4*g``.
    """
    lane = np.arange(64)[:, None]
    field = np.arange(32)[None, :]
    physical = 32 * (lane // 32) + field
    paired = (physical & 0x0F) | ((physical & 0x10) << 1) | ((physical & 0x20) >> 1)
    group = paired // 32
    byte = paired % 32
    return (32 * (byte // 16) + 8 * ((byte % 16) // 4) + byte % 4 + 4 * group).astype(
        np.int32
    )


if _HAVE_TRITON:

    @triton.jit
    def _e2m3_encode_triton(value):
        """Encode scaled FP32 values to signed E2M3 with round-to-nearest-even."""
        magnitude = tl.minimum(tl.abs(value), 7.5)
        magnitude_bits = magnitude.to(tl.int32, bitcast=True)
        rounded_bits = magnitude_bits + 0x7FFFF + ((magnitude_bits >> 20) & 1)
        exponent = ((rounded_bits >> 23) & 0xFF) - 126
        normal_code = (exponent << 3) | ((rounded_bits >> 20) & 7)

        scaled_subnormal = magnitude * 8.0
        floor_value = tl.floor(scaled_subnormal)
        floor_code = floor_value.to(tl.int32)
        fraction = scaled_subnormal - floor_value
        round_up = (fraction > 0.5) | ((fraction == 0.5) & ((floor_code & 1) == 1))
        subnormal_code = floor_code + round_up.to(tl.int32)

        magnitude_code = tl.where(magnitude >= 1.0, normal_code, subnormal_code)
        magnitude_code = tl.minimum(tl.maximum(magnitude_code, 0), 31)
        sign = (value.to(tl.int32, bitcast=True) < 0).to(tl.int32) * 32
        return magnitude_code | sign

    @triton.jit
    def _pack_v_fp6_kernel(
        v_ptr,  # fp8 V [b, sk, h_kv, d] (any strides)
        out_ptr,  # uint8 [b*h_kv*nT*12800]
        scale_ptr,
        kvtab_ptr,  # int32 [64*32] (L*32 + f) -> kv-in-64-chunk offset
        stride_vb,
        stride_vs,
        stride_vh,
        stride_vd,
        sk,
        h_kv,
        nT,
        n_blocks,  # total 32-kv MX blocks
        CLAMP_TAIL: tl.constexpr,
        FIXED_E8M0: tl.constexpr,
        SEPARATE_OUTPUT: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        blk = pid * BLOCK_N + tl.arange(0, BLOCK_N)  # [BN]
        m = blk < n_blocks
        # decode block id: blk = ((bh*nT + t)*128 + d_row)*4 + kvblk
        kvblk = blk % 4
        physical_d = (blk // 4) % 128
        d_row = physical_d
        t = (blk // 512) % nT
        bh = blk // (512 * nT)
        bb = bh // h_kv
        hh = bh % h_kv
        n = physical_d // 32
        k = kvblk // 2
        bn = n * 2 + k
        L = (kvblk % 2) * 32 + (physical_d % 32)

        f = tl.arange(0, 32)
        kt = tl.load(kvtab_ptr + L[:, None] * 32 + f[None, :])  # [BN,32]
        kv = (t * 128 + k * 64)[:, None] + kt  # [BN,32] kv-in-tile
        if CLAMP_TAIL:
            kv = tl.minimum(kv, sk - 1)
        voff = (
            bb[:, None] * stride_vb
            + kv * stride_vs
            + hh[:, None] * stride_vh
            + d_row[:, None] * stride_vd
        )
        vals = tl.load(v_ptr + voff, mask=m[:, None], other=0.0).to(tl.float32)

        amax = tl.max(tl.abs(vals), axis=1)  # [BN]
        bits = amax.to(tl.int32, bitcast=True)
        exp = (bits >> 23) & 0xFF
        E = 0 if FIXED_E8M0 else tl.where(amax == 0.0, 0, exp - 129)
        inv_scale = tl.exp2((-E).to(tl.float32))  # 2^-E (exact dyadic)
        y = vals * inv_scale[:, None]  # scaled (exact in fp32 for fp8 input)
        codes = _e2m3_encode_triton(y)  # [BN,32] field-order 6-bit codes

        cf = codes.reshape(BLOCK_N, 8, 4)
        w = (1 << (6 * tl.arange(0, 4))).to(tl.int32)  # [1,6,12,18] shifts
        u = tl.sum(cf * w[None, None, :], axis=2)  # [BN,8] 24-bit packed words
        b0 = (u & 0xFF).to(tl.uint8)
        b1 = ((u >> 8) & 0xFF).to(tl.uint8)
        b2 = ((u >> 16) & 0xFF).to(tl.uint8)

        base = (bh * nT + t) * (12288 if SEPARATE_OUTPUT else 12800)
        data_off = base + bn * 1536 + L * 24  # [BN]
        g = tl.arange(0, 8)
        off0 = data_off[:, None] + g[None, :] * 3
        tl.store(out_ptr + off0 + 0, b0, mask=m[:, None])
        tl.store(out_ptr + off0 + 1, b1, mask=m[:, None])
        tl.store(out_ptr + off0 + 2, b2, mask=m[:, None])
        sb = ((E + 127) & 0xFF).to(tl.uint8)
        if SEPARATE_OUTPUT:
            scale_base = (bh * nT + t) * 512
            scale_lane = physical_d % 32 + 32 * (kvblk % 2)
            scale_off = (
                scale_base + (kvblk // 2) * 256 + scale_lane * 4 + physical_d // 32
            )
            tl.store(scale_ptr + scale_off, sb, mask=m)
            if pid == 0:
                tail = tl.arange(0, 256)
                tl.store(out_ptr + n_blocks * 24 + tail, 0)
        else:
            scale_off = base + 12288 + physical_d * 4 + kvblk
            tl.store(out_ptr + scale_off, sb, mask=m)


def _qk_field_perm() -> np.ndarray:
    """Per-output-field source index within a 32-block for the lastdim pack.

    Matches quantize_fp6_lastdim's interleave field[2i]=blk[i], field[2i+1]=
    blk[16+i] (no kv scramble), so loading in this order yields fields already in
    packed position."""
    f = np.arange(32)
    return np.where(f % 2 == 0, f // 2, 16 + f // 2).astype(np.int32)


if _HAVE_TRITON:

    @triton.jit
    def _pack_qk_fp6_kernel(
        x_ptr,  # float [N, D] row-major (D % 32 == 0)
        packed_ptr,  # uint8 [N, NB*24]
        scale_ptr,  # uint8 [N, NB]
        cperm_ptr,  # int32 [32] field->source-element permutation
        D,
        NB,  # D // 32
        n_blocks,  # N * NB
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        blk = pid * BLOCK_N + tl.arange(0, BLOCK_N)  # [BN]
        m = blk < n_blocks
        row = blk // NB
        bj = blk % NB  # which 32-block within the last dim

        f = tl.arange(0, 32)
        cp = tl.load(cperm_ptr + f)  # [32]
        elem = bj[:, None] * 32 + cp[None, :]  # [BN,32] source element index
        xoff = row[:, None] * D + elem
        vals = tl.load(x_ptr + xoff, mask=m[:, None], other=0.0).to(tl.float32)

        amax = tl.max(tl.abs(vals), axis=1)  # [BN]
        bits = amax.to(tl.int32, bitcast=True)
        exp = (bits >> 23) & 0xFF
        E = tl.where(amax == 0.0, 0, exp - 129)  # frexp_exp-3
        inv_scale = tl.exp2((-E).to(tl.float32))
        y = vals * inv_scale[:, None]
        codes = _e2m3_encode_triton(y)  # [BN,32] field-order codes

        cf = codes.reshape(BLOCK_N, 8, 4)
        w = (1 << (6 * tl.arange(0, 4))).to(tl.int32)
        u = tl.sum(cf * w[None, None, :], axis=2)  # [BN,8]
        b0 = (u & 0xFF).to(tl.uint8)
        b1 = ((u >> 8) & 0xFF).to(tl.uint8)
        b2 = ((u >> 16) & 0xFF).to(tl.uint8)

        base = row * (NB * 24) + bj * 24  # [BN] byte base in packed
        g = tl.arange(0, 8)
        off0 = base[:, None] + g[None, :] * 3
        tl.store(packed_ptr + off0 + 0, b0, mask=m[:, None])
        tl.store(packed_ptr + off0 + 1, b1, mask=m[:, None])
        tl.store(packed_ptr + off0 + 2, b2, mask=m[:, None])
        scale_off = row * NB + bj
        sb = ((E + 127) & 0xFF).to(tl.uint8)
        tl.store(scale_ptr + scale_off, sb, mask=m)

    @triton.jit
    def _gather_k_lds_kernel(
        packed_ptr,  # uint8 packed K [b, sk, h, 96] flattened (contiguous)
        buf_ptr,  # uint8 LDS-order output buffer [b, h, k_hs] flattened
        srcw_ptr,  # int32 [k_hs] within-(b,h) source byte offset = (gc//96)*(h*96)+(gc%96)
        valid_ptr,  # int8 [k_hs] 1=keep, 0=zero (fp6 dup/overflow + partial-seq tail)
        DATA_HS,  # nt*12288 data bytes per (b,h)
        TILE_BYTES,
        SKH96,  # sk*h*96 = packed bytes per batch
        H,  # heads
        DATA_TILE_BYTES: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # One fused pass replacing the torch permute+contiguous / advanced-index gather /
        # masked_fill / buffer-copy chain (~4 full-size passes -> 1 gathered read + 1 write).
        pid = tl.program_id(0)
        nchunk = DATA_HS // BLOCK
        bh = pid // nchunk
        chunk = pid % nchunk
        bIdx = bh // H
        hIdx = bh % H
        p = chunk * BLOCK + tl.arange(0, BLOCK)
        srcw = tl.load(srcw_ptr + p)
        valid = tl.load(valid_ptr + p)
        src_addr = bIdx * SKH96 + hIdx * 96 + srcw
        byte = tl.load(packed_ptr + src_addr).to(tl.int32)
        byte = tl.where(valid != 0, byte, 0).to(tl.uint8)
        tile = p // DATA_TILE_BYTES
        in_tile = p - tile * DATA_TILE_BYTES
        dst_addr = (
            bh * (DATA_HS // DATA_TILE_BYTES) * TILE_BYTES + tile * TILE_BYTES + in_tile
        )
        tl.store(buf_ptr + dst_addr, byte)

    @triton.jit
    def _fill_k_scale_tail_kernel(
        scale_ptr,  # uint8 scale [b, sk, h, 4] flattened
        buf_ptr,  # uint8 packed K buffer [b,h,nt*17408] flattened
        SK,
        H,
        NT,
        TILE_BYTES: tl.constexpr,
        SCALE_TAIL_OFFSET: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        bh = pid // NT
        t = pid % NT
        bidx = bh // H
        hidx = bh % H
        offs = tl.arange(0, BLOCK)
        region_b = offs >= 512
        region_off = offs - region_b.to(tl.int32) * 512
        inst = region_off >> 8
        lane = (region_off & 255) >> 2
        byte_in_dword = region_off & 3
        src_shift = byte_in_dword + region_b.to(tl.int32)
        src_token = (
            t * 128 + ((lane & 3) << 5) + (lane >> 2) + inst * 16 + (src_shift >> 2)
        )
        src_byte = src_shift & 3
        dst = bh * (NT * TILE_BYTES) + t * TILE_BYTES + SCALE_TAIL_OFFSET + offs
        src = ((bidx * SK + src_token) * H + hidx) * 4 + src_byte
        valid = src_token < SK
        val = tl.load(scale_ptr + src, mask=valid, other=0).to(tl.uint8)
        tl.store(buf_ptr + dst, val)


_QK_FIELD_PERM_CACHE: dict = {}


def _qk_field_perm_dev(device):
    """Cache the last-dimension FP6 field permutation on each device."""
    cperm = _QK_FIELD_PERM_CACHE.get(device)
    if cperm is None:
        cperm = torch.from_numpy(_qk_field_perm()).to(device)
        _QK_FIELD_PERM_CACHE[device] = cperm
    return cperm


def quantize_fp6_lastdim_triton(x: "torch.Tensor"):
    """Quantize the last dimension in 32-value MXFP6 E2M3 blocks."""
    assert _HAVE_TRITON, "triton/torch unavailable"
    *lead, D = x.shape
    assert D % 32 == 0, D
    NB = D // 32
    xc = x.contiguous()
    xflat = xc.reshape(-1, D)
    N = xflat.shape[0]
    packed = torch.empty(N, NB * 24, dtype=torch.uint8, device=x.device)
    scale = torch.empty(N, NB, dtype=torch.uint8, device=x.device)
    cperm = _qk_field_perm_dev(x.device)
    n_blocks = N * NB
    # The hd128 FMHA workloads consistently select 16/1. Pin it to avoid paying and logging
    # the five-config autotune in every fresh benchmark process.
    grid = (triton.cdiv(n_blocks, 16),)
    _pack_qk_fp6_kernel[grid](
        xflat,
        packed,
        scale,
        cperm,
        D,
        NB,
        n_blocks,
        BLOCK_N=16,
        num_warps=1,
    )
    return (
        packed.reshape(*lead, NB * 24),
        scale.reshape(*lead, NB),
    )


_K_LDS_GIDX_CACHE: dict = {}


def _k_lds_gather_index(nt: int, total: int, device):
    """Cached [(nt*12288)] compact LDS-order gather index + valid mask. valid = (g < total)
    zeroes BOTH the fp6 dup/overflow LDS tail AND a partial seq tail. Keyed by
    (nt, total, device): with a partial tail the same nt pairs with different total."""
    key = (nt, total, device)
    g = _K_LDS_GIDX_CACHE.get(key)
    if g is None:
        idx16k = torch.as_tensor(
            _k_lds_order_gather_index(), dtype=torch.long, device=device
        )
        full = (
            torch.arange(nt, device=device, dtype=torch.long) * _K_COMPACT_DATA_BYTES
        ).unsqueeze(1) + idx16k.unsqueeze(0)
        full = full.reshape(-1)
        valid = full < total
        gc = torch.where(valid, full, torch.zeros_like(full))
        g = (gc, valid)
        _K_LDS_GIDX_CACHE[key] = g
    return g


_K_LDS_SRCW_CACHE: dict = {}


def _k_lds_src_within(nt: int, total: int, h: int, device):
    """Cached (srcw int32 [k_hs], valid int8 [k_hs]) for the fused LDS gather kernel.
    srcw = (gc//96)*(h*96) + (gc%96) folds the [b,sk,h,96]->[b,h,token-major] permute
    into the source address so the kernel reads `packed` directly (no permute+contiguous
    copy). h-dependent (the h*96 token stride) so h is part of the key."""
    key = (nt, total, h, device)
    g = _K_LDS_SRCW_CACHE.get(key)
    if g is None:
        gc, valid = _k_lds_gather_index(nt, total, device)
        srcw = ((gc // 96) * (h * 96) + (gc % 96)).to(torch.int32)
        g = (srcw, valid.to(torch.int8))
        _K_LDS_SRCW_CACHE[key] = g
    return g


def reorder_fp6_k_lds_order_triton(
    packed: "torch.Tensor",
    scale: "torch.Tensor",
    tile: int = 128,
    return_raw: bool = False,
):
    """Reorder dense packed K into the kernel-ready LDS-order view WITH E8M0 scales in the
    per-tile tail. Each 128-token tile retains its 17408B global ABI: 12288B compact chunk-major
    fp6 K data + a 4096B unused hole + a 1024B lane-major K-scale image. The kernel loads the
    scale straight from the K buffer tail (coalesced buffer_load lds:1), so there is no separate
    K-scale global-load stream. Supports S % tile != 0 (the gather's valid mask zeroes the partial
    tail tile, which the kernel masks in softmax).

    packed : dense uint8 fp6 K [b, sk, h, 96] on GPU.
    scale : dense uint8 E8M0 K scales [b, sk, h, 4] on GPU.
    Returns (k_view uint8 [b, sk, h, 96] strided (seq stride 136) over a [b,h,n_tiles*17408]
             buffer, scale uint8 [b, sk, h, 4]). `scale` only satisfies the k_descale ABI arg --
             the kernel reads scales from the K tail, not this tensor.
    If return_raw: returns (buf, sbuf) -- the FULL contiguous backing buffers (uint8 1D) instead of
    the strided/padded views. A torch.library.custom_op caller MUST take this path: returning the
    strided k_view as a custom-op output lets AOTAutograd clone it to a contiguous numel-sized
    tensor (dropping the seq-stride-136 LDS layout -> garbage). The caller rebuilds
    k_view = buf.as_strided((b, sk, h, 96), (h*nt*17408, 136, nt*17408, 1)) OUTSIDE the op.
    """
    assert _HAVE_TRITON, "triton/torch unavailable"
    b, sk, h, packed_d = packed.shape
    assert packed_d == _K_PACKED_ROW_BYTES and tile == 128, (packed_d, sk, tile)
    assert scale.shape == (b, sk, h, 4), scale.shape
    assert packed.dtype == torch.uint8 and scale.dtype == torch.uint8, (
        packed.dtype,
        scale.dtype,
    )
    assert packed.device == scale.device, (packed.device, scale.device)
    packed = packed.contiguous()
    scale = scale.contiguous()
    nt = (sk + tile - 1) // tile  # ceil; partial tail handled by the valid mask
    total = sk * 96
    # Fused on-device LDS reorder: a single Triton gather (read `packed` via the cached
    # source offset, apply the valid mask, write the buffer) replaces the torch chain of
    # permute+contiguous / advanced-index gather / masked_fill / buffer-copy (~4 full-size
    # passes -> 1 gathered read + 1 write; ~1.35x faster on the K reorder at long seq).
    srcw, valid8 = _k_lds_src_within(nt, total, h, packed.device)
    # Each 128-token tile remains 17408B: 12288B compact fp6 K data, a 4096B unused hole, then the
    # 1024B lane-major E8M0 K-scale tail.
    # The kernel loads that scale image with a coalesced buffer_load lds:1 straight from the K
    # buffer (no separate scale pointer / global_load) -- this removes the stalling K-scale global
    # loads. seq stride = 136 (17408/128) -> the kernel's _s_k_Seqs=136 -> tile base = token*136.
    k_tile_bytes = _K_TILE_BYTES
    k_hs = nt * k_tile_bytes
    k_bs = h * k_hs
    buf = torch.empty(b * k_bs + 256, dtype=torch.uint8, device=packed.device)
    BLOCK = 1024
    data_hs = nt * _K_COMPACT_DATA_BYTES
    assert data_hs % BLOCK == 0, (data_hs, BLOCK)
    grid = (b * h * (data_hs // BLOCK),)
    _gather_k_lds_kernel[grid](
        packed.reshape(-1),
        buf,
        srcw,
        valid8,
        data_hs,
        k_tile_bytes,
        sk * h * 96,
        h,
        DATA_TILE_BYTES=_K_COMPACT_DATA_BYTES,
        BLOCK=BLOCK,
        num_warps=4,
    )
    # Fill the per-tile 1024B scale tail: Region A (unshifted) + Region B (pre-shifted +1 byte, so
    # the kernel MFMA op_sel picks dblk1/dblk3 with no runtime shift). The B pre-shift reads 1 byte
    # past the last token's scale on the final tile -> the +256 buf slack keeps it mapped.
    _fill_k_scale_tail_kernel[(b * h * nt,)](
        scale.reshape(-1),
        buf,
        sk,
        h,
        nt,
        TILE_BYTES=_K_TILE_BYTES,
        SCALE_TAIL_OFFSET=_K_SCALE_TAIL_OFFSET,
        BLOCK=1024,
        num_warps=4,
    )
    k_view = buf.as_strided(
        (b, sk, h, _K_PACKED_ROW_BYTES),
        (k_bs, _K_SEQ_STRIDE_BYTES, k_hs, 1),
    )
    # `scale` is still returned to satisfy the k_descale ABI arg, but the kernel reads scales from
    # the K tail, not this tensor. Re-home into a +64 slack buffer (harmless; keeps callers happy).
    sflat = scale.reshape(-1)
    sbuf = torch.empty(sflat.numel() + 64, dtype=torch.uint8, device=scale.device)
    sbuf[: sflat.numel()] = sflat
    if return_raw:
        return buf, sbuf
    scale = sbuf[: sflat.numel()].view(b, sk, h, 4)
    return k_view, scale


def quantize_fp6_k_lds_order_triton(
    k_thd: "torch.Tensor", tile: int = 128, return_raw: bool = False
):
    """Quantize float K and reorder it into the kernel-ready LDS-order fp6 view.

    Use ``quantize_fp6_lastdim_triton`` followed by ``reorder_fp6_k_lds_order_triton`` when dense
    quantization should be scheduled independently from the kernel-specific LDS layout conversion.
    """
    _b, sk, _h, d = k_thd.shape
    assert d == 128 and tile == 128, (d, sk, tile)
    packed, scale = quantize_fp6_lastdim_triton(k_thd)
    return reorder_fp6_k_lds_order_triton(
        packed, scale, tile=tile, return_raw=return_raw
    )


def fp6_k_lds_order_views_from_raw(
    buf: "torch.Tensor",
    sbuf: "torch.Tensor",
    b: int,
    sk: int,
    h: int,
    tile: int = 128,
):
    """Rebuild the mxfp6 kernel ABI views from contiguous direct-packer buffers."""
    assert tile == 128, tile
    nt = (sk + tile - 1) // tile
    k_hs = nt * _K_TILE_BYTES
    k_bs = h * k_hs
    k_view = buf.as_strided(
        (b, sk, h, _K_PACKED_ROW_BYTES),
        (k_bs, _K_SEQ_STRIDE_BYTES, k_hs, 1),
    )
    scale = sbuf[: b * sk * h * 4].view(b, sk, h, 4)
    return k_view, scale


_QK_FIELD_PERM_PT_CACHE: dict = {}


def _qk_field_perm_pt(device):
    """Cached int64 field permutation [32] for the torch lastdim fp6 pack (same perm as the
    Triton _qk_field_perm). Built once per device so it is not rebuilt in a capture region.
    """
    p = _QK_FIELD_PERM_PT_CACHE.get(device)
    if p is None:
        p = torch.as_tensor(_qk_field_perm().astype(np.int64), device=device)
        _QK_FIELD_PERM_PT_CACHE[device] = p
    return p


def _e2m3_encode_torch(y: "torch.Tensor") -> "torch.Tensor":
    """Branchless round-half-even E2M3 encode (torch port of the _pack_qk_fp6_kernel encode).
    y float32 [...] -> uint8 codes [...] (0..63; bit5 = sign). Same normal (fp32 RNE round to 3
    mantissa bits) / subnormal (round(mag*8)) split + tie-to-even as the Triton kernel.
    """
    mag = y.abs().clamp(max=7.5)
    magbits = mag.contiguous().view(torch.int32)
    bits_r = magbits + 0x7FFFF + ((magbits >> 20) & 1)
    exp2 = ((bits_r >> 23) & 0xFF) - 126
    m3n = (bits_r >> 20) & 7
    code_norm = (exp2 << 3) | m3n
    t8 = mag * 8.0
    fl = torch.floor(t8)
    fli = fl.to(torch.int32)
    frac = t8 - fl
    up = (frac > 0.5) | ((frac == 0.5) & ((fli & 1) == 1))
    code_sub = fli + up.to(torch.int32)
    chosen = torch.where(mag >= 1.0, code_norm, code_sub).clamp(0, 31)
    sign = (y.contiguous().view(torch.int32) < 0).to(torch.int32) * 32
    return (chosen | sign).to(torch.uint8)


def quantize_fp6_lastdim_torch(x: "torch.Tensor"):
    """Torch-compile-friendly last-dimension MXFP6 E2M3 quantization."""
    assert _HAVE_TRITON, "torch unavailable"
    lead = list(x.shape[:-1])
    D = x.shape[-1]
    assert D % 32 == 0, D
    NB = D // 32
    xf = x.to(torch.float32).reshape(*lead, NB, 32)
    amax = xf.abs().amax(dim=-1)  # [..., NB]
    bits = amax.contiguous().view(torch.int32)
    exp = (bits >> 23) & 0xFF
    E = torch.where(amax == 0, torch.zeros_like(exp), exp - 129)  # frexp_exp - 3
    inv_scale = torch.exp2((-E).to(torch.float32))  # 2^-E (exact dyadic)
    cperm = _qk_field_perm_pt(x.device)
    y = xf.index_select(-1, cperm) * inv_scale.unsqueeze(-1)  # field-order, scaled
    codes = _e2m3_encode_torch(y)  # [..., NB, 32] uint8
    # pack 32 six-bit fields -> 24 bytes (groups of 4 fields = 24 bits = 3 bytes).
    c = codes.to(torch.int32).reshape(*lead, NB, 8, 4)
    u = (
        c[..., 0] | (c[..., 1] << 6) | (c[..., 2] << 12) | (c[..., 3] << 18)
    )  # [..., NB, 8]
    packed = (
        torch.stack([u & 0xFF, (u >> 8) & 0xFF, (u >> 16) & 0xFF], dim=-1)
        .to(torch.uint8)
        .reshape(*lead, NB * 24)
    )
    scale = ((E + 127) & 0xFF).to(torch.uint8)
    return packed, scale


_K_SCALE_TAIL_IDX_CACHE: dict = {}


def _k_scale_tail_index(nt: int, sk: int, h: int, device):
    """Cached (sidx int64 [h, nt, 1024], valid bool [nt, 1024]) for the per-tile K-scale TAIL image
    (torch port of _fill_k_scale_tail_kernel: Region A unshifted + Region B pre-shifted +1 byte).
    sidx indexes the flat [sk*h*4] E8M0 scale (per batch) = tok*(h*4) + head*4 + byte; invalid
    (pre-shift tail past sk) -> clamped to 0 and masked out."""
    key = (nt, sk, h, device)
    g = _K_SCALE_TAIL_IDX_CACHE.get(key)
    if g is None:
        offs = torch.arange(1024, device=device, dtype=torch.int64)
        region_b = (offs >= 512).to(torch.int64)
        region_off = offs - region_b * 512
        inst = region_off >> 8
        lane = (region_off & 255) >> 2
        byte_in_dword = region_off & 3
        src_shift = byte_in_dword + region_b
        tok_local = (
            ((lane & 3) << 5) + (lane >> 2) + inst * 16 + (src_shift >> 2)
        )  # [1024]
        src_byte = src_shift & 3  # [1024]
        t = torch.arange(nt, device=device, dtype=torch.int64)
        src_token = t[:, None] * 128 + tok_local[None, :]  # [nt, 1024]
        valid = src_token < sk
        hidx = torch.arange(h, device=device, dtype=torch.int64)
        sidx = (
            src_token[None] * (h * 4)
            + hidx[:, None, None] * 4
            + src_byte[None, None, :]
        )
        sidx = torch.where(valid[None], sidx, torch.zeros_like(sidx))  # [h, nt, 1024]
        g = (sidx, valid)
        _K_SCALE_TAIL_IDX_CACHE[key] = g
    return g


def quantize_fp6_k_lds_order_torch(
    k_thd: "torch.Tensor", tile: int = 128, return_raw: bool = False
):
    """Graph-friendly (pure-torch) port of quantize_fp6_k_lds_order_triton (identical 17408B/tile
    ABI: 12288B compact fp6 K data + 4096B unused + 1024B lane-major E8M0 K-scale tail). Traceable by
    Inductor (torch pack + index-gathers + cat) so K packing can overlap distributed communication.
    Byte-identical to the Triton packer (reuses the exact LDS gather / scale-tail index tables).

    k_thd float K [b, sk, h, 128] -> (k_view uint8 [b, sk, h, 96] strided (seq stride 136) over a
    [b, h, nt*17408] buffer, scale uint8 [b, sk, h, 4] (ABI only; the kernel reads scales from the
    K tail)). If return_raw: (buf, sbuf) contiguous backing buffers (for a torch.library.custom_op
    caller that must rebuild the strided view outside the op)."""
    assert _HAVE_TRITON, "torch unavailable"
    b, sk, h, d = k_thd.shape
    assert d == 128 and tile == 128, (d, sk, tile)
    nt = (sk + tile - 1) // tile  # ceil; the valid mask zeroes a partial tail tile
    packed, scale = quantize_fp6_lastdim_torch(k_thd)  # [b,sk,h,96], [b,sk,h,4]
    total = sk * 96

    # DATA region: token-major per head, then the LDS-order gather (shared across heads), invalid->0.
    km = packed.permute(0, 2, 1, 3).reshape(b, h, sk * 96).contiguous()
    gc, dvalid = _k_lds_gather_index(nt, total, k_thd.device)  # compact data indices
    data = km[:, :, gc]
    data = torch.where(dvalid[None, None, :], data, torch.zeros_like(data)).reshape(
        b, h, nt, _K_COMPACT_DATA_BYTES
    )

    # SCALE-TAIL region (1024B/tile): gather the E8M0 scale into the lane-major tail image, invalid->0.
    sidx, svalid = _k_scale_tail_index(nt, sk, h, k_thd.device)
    sf = scale.reshape(b, sk * h * 4)
    stail = sf[:, sidx.reshape(-1)].reshape(b, h, nt, 1024)
    stail = torch.where(svalid[None, None], stail, torch.zeros_like(stail))

    # Preserve the 17408B global tile ABI: compact data + unused staging hole + scale tail.
    padding = data.new_zeros(b, h, nt, _K_RESERVED_BYTES)
    buf_full = torch.cat([data, padding, stail], dim=-1)  # [b, h, nt, 17408]
    k_tile_bytes = _K_TILE_BYTES
    k_hs = nt * k_tile_bytes
    k_bs = h * k_hs
    buf = torch.cat([buf_full.reshape(-1), buf_full.new_zeros(256)])
    sflat = scale.reshape(-1)
    sbuf = torch.cat([sflat, sflat.new_zeros(64)])
    if return_raw:
        return buf, sbuf
    k_view = buf.as_strided(
        (b, sk, h, _K_PACKED_ROW_BYTES),
        (k_bs, _K_SEQ_STRIDE_BYTES, k_hs, 1),
    )
    scale_out = sbuf[: sflat.numel()].view(b, sk, h, 4)
    return k_view, scale_out


def pack_fp6_v_data_scale_views(
    v: "torch.Tensor", tile: int = 128, fixed_e8m0: bool = False
):
    """Pack V into separate F8F6 data and E8M0 scale images."""
    assert _HAVE_TRITON, "triton/torch unavailable"
    b, sk, h_kv, d = v.shape
    n_tiles = (sk + tile - 1) // tile

    data_flat, scale_flat = quantize_fp6_v_data_scale_triton(
        v, tile=tile, fixed_e8m0=fixed_e8m0
    )
    v_hs = n_tiles * 12288
    v_bs = h_kv * v_hs
    view = data_flat.as_strided((b, sk, h_kv, d), (v_bs, 96, v_hs, 1))
    return view, scale_flat.view(b, h_kv, n_tiles * 512)
