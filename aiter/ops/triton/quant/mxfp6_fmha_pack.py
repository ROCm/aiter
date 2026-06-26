# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Host MXFP6-E2M3 packers for the fp6 FMHA (Sage-attention) gfx950 kernel.

Self-contained host-side numpy packers that cast Q/K/V to the exact MXFP6-E2M3
byte layout the ``fwd_hd128_mxfp6`` kernel consumes (no in-kernel re-quant). This
module is the canonical, production home of the fp6 FMHA encoding logic; it is
INDEPENDENT of the mxfp4 path and shares no state with it.

Only the PROVEN layouts are kept here (cos >= 0.99 @ b1 hq5 sq256 seed0, == the
in-kernel "both"-mode reference). The experimental layout zoo used during bring-up
lives in the research repo and is reachable from the benchmark via the
``AITER_MXFP6_PACK`` path override.

Layout facts (all measured / proven on gfx950):

  * E2M3 6-bit code = OCP MXFP6 "S EE MMM": bit5=sign, bits4:3=exp(bias 1),
    bits2:0=mantissa, subnormals at exp==0. Full 32-level grid (max 7.5).
  * Per 32-element MX block: E8M0 scale exponent E = frexp_exp(amax) - 3
    (== floor(log2(amax)) - emax, emax(E2M3)=2). Scale byte = E + 127. Each
    value v is stored as code(v / 2^E).
  * 24-byte (6-dword) block, 6-bit fields LSB-first at bit f*6. The MFMA reads
    field 2i = blk[i], field 2i+1 = blk[16+i] (interleaved).
"""

import os

import numpy as np

try:
    import torch
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except ImportError:  # numpy-only host packing still works without triton/torch
    _HAVE_TRITON = False


# ---------------------------------------------------------------------------
# E2M3 grid + scalar encode
# ---------------------------------------------------------------------------
def _build_e2m3_grid() -> np.ndarray:
    """OCP MXFP6 E2M3 magnitude table: code (0..31) -> magnitude (ascending)."""
    g = np.empty(32, dtype=np.float64)
    for code in range(32):
        exp = code >> 3
        m = code & 7
        g[code] = (m / 8.0) if exp == 0 else (2.0 ** (exp - 1)) * (1.0 + m / 8.0)
    return g


_E2M3_MAG = _build_e2m3_grid()  # index == 6-bit code (sans sign); ascending
_FP6_ROUND = os.environ.get("MXFP4_FP6_ROUND", "rne")  # rne|rtz|rhu


def e2m3_encode(x: np.ndarray) -> np.ndarray:
    """Round-encode f32 -> 6-bit E2M3 code (uint8, 0..63). Mode via MXFP4_FP6_ROUND
    (rne=round-half-even default, rtz=truncate toward zero, rhu=round-half-up)."""
    x = np.asarray(x, dtype=np.float64)
    sign = (x < 0) | ((x == 0) & (np.signbit(x)))
    mag = np.abs(x)
    grid = _E2M3_MAG  # ascending, code == index
    mag = np.minimum(mag, grid[-1])  # clamp to max 7.5
    idx = np.searchsorted(grid, mag, side="left")
    idx = np.clip(idx, 0, len(grid) - 1)
    lo = np.clip(idx - 1, 0, len(grid) - 1)
    dlo = mag - grid[lo]
    dhi = grid[idx] - mag
    if _FP6_ROUND == "rtz":
        chosen = lo  # truncate toward zero (lo is always <= mag)
    elif _FP6_ROUND == "rhu":
        chosen = np.where(dhi <= dlo, idx, lo)  # round-half-up (toward +inf mag)
    else:  # rne
        pick_hi = dhi < dlo
        tie = dhi == dlo
        pick_hi = pick_hi | (tie & ((lo % 2) == 1))
        chosen = np.where(pick_hi, idx, lo)
    code = chosen.astype(np.uint8)
    code = np.where(sign, code | 0x20, code).astype(np.uint8)
    return code


def e2m3_decode(code: np.ndarray) -> np.ndarray:
    """Decode 6-bit E2M3 code -> f32 magnitude*sign (verification helper)."""
    code = np.asarray(code, dtype=np.uint8)
    sign = (code & 0x20) != 0
    mag = _E2M3_MAG[(code & 0x1F)]
    return np.where(sign, -mag, mag).astype(np.float64)


# ---------------------------------------------------------------------------
# Fast 6-bit field packing
# ---------------------------------------------------------------------------
def _pack_fields_24b(fields: np.ndarray) -> np.ndarray:
    """Pack [..., 32] of 6-bit codes LSB-first into [..., 24] bytes (vectorized).

    32 fp6 fields = 192 bits = 24 bytes. Each group of 4 consecutive fields spans
    exactly 24 bits = 3 byte-aligned bytes, so pack 4 codes into a uint32 (field i
    at bit 6i) and emit the low 3 little-endian bytes. Byte-identical to the naive
    per-bit loop, ~13x faster (no 192-iteration python loop)."""
    f = fields.reshape(-1, 8, 4).astype(np.uint32)
    v = f[..., 0] | (f[..., 1] << 6) | (f[..., 2] << 12) | (f[..., 3] << 18)  # [N,8]
    b = np.stack([v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF], axis=-1).astype(
        np.uint8
    )  # [N,8,3]
    return b.reshape(*fields.shape[:-1], 24)


# ---------------------------------------------------------------------------
# QK packer (and the V operand building block)
# ---------------------------------------------------------------------------
def quantize_fp6_lastdim(x: np.ndarray):
    """Vectorized MXFP6-E2M3 quantize along the last dim (multiple of 32).

    x: f32 array [..., D], D % 32 == 0.
    Returns:
      packed: uint8 [..., (D//32)*24]  (24 bytes per 32-block, interleaved fields)
      scale:  uint8 [..., D//32]       (E8M0 = E+127 per block)
    Mirrors the kernel/HW fp6 pack: E = frexp_exp(amax)-3, value -> code(v/2^E),
    field[2i]=code(blk[i]/2^E), field[2i+1]=code(blk[16+i]/2^E)."""
    x = np.asarray(x, dtype=np.float64)
    *lead, D = x.shape
    assert D % 32 == 0, D
    nb = D // 32
    blk = x.reshape(*lead, nb, 32)
    amax = np.max(np.abs(blk), axis=-1)  # [..., nb]
    _m, e = np.frexp(np.maximum(amax, np.float64(0)))
    E = np.where(amax == 0, 0, e - 3).astype(np.int64)  # [..., nb]
    scale = (2.0**E)[..., None]  # [..., nb, 1]
    codes = e2m3_encode(blk / scale)  # [..., nb, 32] uint8
    # interleave -> field order: field[2i]=blk[i], field[2i+1]=blk[16+i]
    fields = np.empty_like(codes)
    fields[..., 0::2] = codes[..., 0:16]
    fields[..., 1::2] = codes[..., 16:32]
    packed = _pack_fields_24b(fields).reshape(*lead, nb * 24)
    scale_b = ((E + 127) & 0xFF).astype(np.uint8)
    return packed, scale_b


# ---------------------------------------------------------------------------
# K LDS-ORDER packer for the COALESCED cooperative load
# ---------------------------------------------------------------------------
# The default kernel cooperative K load reads token-strided (lane v0 -> token
# (v0&31) at the 96B token stride), so each lane's 16B falls in its own cache
# line (~25% L1 coalescing) -> the vL1D address-gen serializes under fp6's load
# volume (the long vmcnt(0) wait). This packer PRE-ARRANGES K so the cooperative
# load is a CONTIGUOUS, coalesced copy that lands byte-identically in the kernel's
# chunk-major LDS image -- so the kernel's _K_COALESCED_LOAD path uses a plain
# contiguous load and lds_read_K_data / the MFMA are unchanged. The transpose
# becomes this one-time host op. (Stalled-on-Address 18.5%->0.7%, L1-L2 txns
# 332M->241M, +1.3% end-to-end vs the token-strided load.)
#
# Permutation derived from the kernel's default chunk-major load addressing:
#   tile (16384 B) LDS/HBM position P = w*1024 + i*4096 + v0*16 + byte  <-  the
#   token-major byte v_K_base(v0)+C_i(w,i)+byte, with
#   v_K_base = (v0&31)*96 + ((v0>>5)&1)*24  and  C_i: blk=w+(i&1)*4; half=blk&1;
#   n=blk>>1; chunk=i>>1; C_i = n*32*96 + half*48 + chunk*16.
def _k_lds_order_gather_index():
    """Per-tile [16384] token-major byte index for each contiguous LDS-order position P."""
    P = np.arange(16384)
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


def quantize_fp6_k_lds_order(k_thd: np.ndarray, tile: int = 128):
    """Pack K -> LDS-order fp6 (for the kernel's _K_COALESCED_LOAD contiguous load) + per-(tok,32)
    E8M0 scale. Numerically identical to quantize_fp6_lastdim; LAYOUT change only.

    Input : k_thd f32 [b, sk, h, 128].
    Output:
      data  uint8 [b, h, n_tiles*16384]  (each tile = the 16384B chunk-major LDS image; the kernel's
            contiguous coalesced load lands it byte-identically to the token-strided chunk-major load).
            tile = 16384B over 128 tokens -> the kernel must see this view with seq stride 128.
      scale uint8 [b, sk, h, 4]
    """
    k = np.asarray(k_thd)
    b, sk, h, d = k.shape
    assert d == 128 and tile == 128, (d, sk, tile)
    nt = (sk + tile - 1) // tile  # ceil; the valid=(g<total) mask zeroes a partial tail tile
    packed, scale = quantize_fp6_lastdim(k.astype(np.float64))  # [b,sk,h,96], [b,sk,h,4]
    # token-major flat per (b,h): [b, h, sk*96]
    km = np.ascontiguousarray(np.transpose(packed, (0, 2, 1, 3))).reshape(b, h, sk * 96)
    idx = _k_lds_order_gather_index()  # [16384]
    total = sk * 96
    out = np.zeros((b, h, nt * 16384), np.uint8)
    for t in range(nt):
        g = t * 12288 + idx
        valid = g < total
        out[:, :, t * 16384:(t + 1) * 16384] = np.where(valid, km[:, :, np.where(valid, g, 0)], 0)
    return np.ascontiguousarray(out).astype(np.uint8), scale.astype(np.uint8)


# ---------------------------------------------------------------------------
# V operand packer (proven "clean" / "operand" layout)
# ---------------------------------------------------------------------------
# tr8 within-32-block kv scramble (4-element chunk / 16-stride interleave). This
# MEASURED permutation makes the host V operand agree with the in-kernel P operand
# (fp8 K-distribution + cvt interleave); without it the layout caps at cos 0.59.
_TR8_SIGMA32 = np.array(
    [
        0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23,
        8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31,
    ],
    dtype=np.int64,
)


def quantize_fp6_v_clean(v_dmajor: np.ndarray, tile: int = 128):
    """Pack V into the per-lane fp6 MFMA operand bytes (PROVEN, cos 0.99682498).

    There is a single proven V layout: V is packed to match the kernel's NATURAL
    (pre-swap) P operand, so the PV MFMA needs no cross-lane permlane32 swap on P
    (the contraction sum_kv P[kv] V[kv] is permutation-invariant over K). The
    field->kv map is the closed form
        kv = t*128 + 64*(bn%2) + kvtab[L, f],
    where kvtab[L,f] = 32*(srcL//32) + fperm[srcF] (see _v_noswap_kvtab); the head
    dim is swap-invariant, d = (bn//2)*32 + (L%32). Per-block E8M0 is computed over
    the gathered 32 kv and written at 12288 + n*128 + (L%32)*4 + (L//32) + 2*k.

    Input : v_dmajor f32 [..., D=128, S] (head dim D on axis -2, kv seq S on -1;
            RAW fp8 magnitudes -- per-channel v_descale is applied in the kernel
            epilogue, so this is numerically a layout change only).
    Output: uint8 [..., n_tiles*(tile*96 + D*4)]. Per 128-kv tile (12800B):
              data  12288B = 8 blocks (n*2+k) x 64 lanes x 24B at (n*2+k)*1536+L*24
              scale   512B = E8M0 at 12288 + n*128 + (L%32)*4 + (L//32) + 2*k.
    """
    v = np.asarray(v_dmajor, dtype=np.float64)
    *lead, D, S = v.shape
    assert D == 128 and S % tile == 0 and tile == 128, (D, S, tile)
    nT = S // tile
    kSubN1, kSubK1 = 4, 2
    nblk = kSubN1 * kSubK1  # 8
    B = int(np.prod(lead)) if lead else 1
    vflat = v.reshape(B, D, S)

    # closed-form pre-swap field->(d,kv) gather (verified == the empirical clean
    # map composed with the kernel's field-level permlane32 swap).
    kvtab = _v_noswap_kvtab()  # [64,32] = 32*(srcL//32) + fperm[srcF]
    bn = np.arange(nblk)
    k_bn = (bn % kSubK1)[:, None, None]  # bn%2
    n_of = (bn // kSubK1)[:, None, None]
    kv_in = 64 * k_bn + kvtab[None]  # [8,64,32] kv-in-tile (pre-swap)
    Lg = np.arange(64)[None, :, None]
    d_in = np.broadcast_to(n_of * 32 + (Lg % 32), (nblk, 64, 32))  # swap-invariant

    # scale byte index (within the 512B region): n*128 + (L%32)*4 + (L//32) + 2*k.
    nn = (bn // kSubK1)[:, None]
    kk = (bn % kSubK1)[:, None]
    LL = np.arange(64)[None, :]
    sidx = (nn * 128 + (LL % 32) * 4 + (LL // 32) + 2 * kk).reshape(-1)  # (512,)

    tile_bytes = tile * 96 + D * 4  # 12800
    out = np.zeros((B, nT * tile_bytes), np.uint8)
    for t in range(nT):
        kvt = t * tile + kv_in  # [8,64,32] absolute kv
        vals = vflat[:, d_in, kvt]  # (B,8,64,32)
        amax = np.max(np.abs(vals), axis=-1)  # (B,8,64)
        _m, e = np.frexp(np.maximum(amax, np.float64(0)))
        E = np.where(amax == 0, 0, e - 3).astype(np.int64)  # (B,8,64)
        codes = e2m3_encode(vals / (2.0**E)[..., None])  # (B,8,64,32)
        data = _pack_fields_24b(codes.reshape(B * nblk * 64, 32))  # (B*8*64,24)
        base = t * tile_bytes
        out[:, base : base + nblk * 64 * 24] = data.reshape(B, nblk * 64 * 24)
        E8 = ((E + 127) & 0xFF).astype(np.uint8).reshape(B, -1)  # (B,512) (bn,L)
        out[:, base + 12288 + sidx] = E8
    return np.ascontiguousarray(out).astype(np.uint8)


# Single proven V layout (the kernel skips the cross-lane P swap), so the historic
# "noswap" / "operand" names all denote this one packer.
quantize_fp6_v_noswap = quantize_fp6_v_clean
quantize_fp6_v_operand_tileflat = quantize_fp6_v_clean


# ---------------------------------------------------------------------------
# Triton GPU V packer (eliminates the one-time host pack)
# ---------------------------------------------------------------------------
# E2M3 magnitude grid as python literals (code 0..31 -> ascending magnitude); the
# Triton kernel reconstructs searchsorted/RNE against these compile-time constants.
_E2M3_GRID = tuple(float(x) for x in _E2M3_MAG)


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


def quantize_fp6_v_clean_triton(v_fp8: "torch.Tensor", tile: int = 128):
    """GPU (Triton) equivalent of quantize_fp6_v_clean (byte-identical).

    v_fp8 : torch fp8 tensor [b, sk, h_kv, d=128] (RAW fp8 V magnitudes; the kernel
            epilogue applies the per-channel descale, so this is a layout cast).
    Returns: torch uint8 [b, h_kv, nT*12800] on the V device, byte-identical to the
    numpy quantize_fp6_v_clean output (all intermediate quantities are exact dyadic
    rationals representable in fp32, so fp32 GPU == fp64 host)."""
    assert _HAVE_TRITON, "triton/torch unavailable"
    b, sk, h_kv, d = v_fp8.shape
    assert d == 128 and tile == 128 and sk % tile == 0, (d, sk, tile)
    nT = sk // tile
    n_blocks = b * h_kv * nT * 128 * 4
    out = torch.empty(b * h_kv * nT * 12800, dtype=torch.uint8, device=v_fp8.device)
    kvtab = torch.from_numpy(_v_noswap_kvtab().reshape(-1)).to(v_fp8.device)
    BLOCK_N = 128
    grid = (triton.cdiv(n_blocks, BLOCK_N),)
    _pack_v_fp6_kernel[grid](
        v_fp8,
        out,
        kvtab,
        v_fp8.stride(0),
        v_fp8.stride(1),
        v_fp8.stride(2),
        v_fp8.stride(3),
        h_kv,
        nT,
        n_blocks,
        GRID=_E2M3_GRID,
        BLOCK_N=BLOCK_N,
    )
    return out.view(b, h_kv, nT * 12800)


# ---------------------------------------------------------------------------
# Triton V packer kv-gather table (pre-swap P operand layout)
# ---------------------------------------------------------------------------
_NOSWAP_KVTAB_CACHE = None


def _v_noswap_kvtab() -> np.ndarray:
    """Per-(lane,field) kv-in-64-chunk offset for the noswap V operand: kv =
    t*128 + 64*k + kvtab[L,f]. Derived from the empirical clean map composed with
    the kernel's field-level permlane32 swap (see quantize_fp6_v_noswap). The
    clean map has the closed form kv = 64*(bn%2) + 32*(L//32) + fperm[f], so
    kvtab[L,f] = 32*(srcL[L,f]//32) + fperm[srcF[L,f]]. Memoized int32 [64,32]."""
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


if _HAVE_TRITON:

    @triton.jit
    def _pack_v_fp6_kernel(
        v_ptr,  # fp8 V [b, sk, h_kv, d] (any strides)
        out_ptr,  # uint8 [b*h_kv*nT*12800]
        kvtab_ptr,  # int32 [64*32] (L*32 + f) -> kv-in-64-chunk offset
        stride_vb,
        stride_vs,
        stride_vh,
        stride_vd,
        h_kv,
        nT,
        n_blocks,  # total 32-kv MX blocks
        GRID: tl.constexpr,  # 32 e2m3 magnitudes (ascending)
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        blk = pid * BLOCK_N + tl.arange(0, BLOCK_N)  # [BN]
        m = blk < n_blocks
        # decode block id: blk = ((bh*nT + t)*128 + d_row)*4 + kvblk
        kvblk = blk % 4
        d_row = (blk // 4) % 128
        t = (blk // 512) % nT
        bh = blk // (512 * nT)
        bb = bh // h_kv
        hh = bh % h_kv
        n = d_row // 32
        k = kvblk // 2
        bn = n * 2 + k
        L = (kvblk % 2) * 32 + (d_row % 32)

        f = tl.arange(0, 32)
        kt = tl.load(kvtab_ptr + L[:, None] * 32 + f[None, :])  # [BN,32]
        kv = (t * 128 + k * 64)[:, None] + kt  # [BN,32] kv-in-tile
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
        E = tl.where(amax == 0.0, 0, exp - 129)  # frexp_exp-3 = (exp-126)-3
        inv_scale = tl.exp2((-E).to(tl.float32))  # 2^-E (exact dyadic)
        y = vals * inv_scale[:, None]  # scaled (exact in fp32 for fp8 input)
        mag = tl.abs(y)
        mag = tl.minimum(mag, 7.5)  # clamp to grid max

        idx = tl.zeros([BLOCK_N, 32], tl.int32)
        glo = tl.full([BLOCK_N, 32], -1.0e30, tl.float32)
        ghi = tl.full([BLOCK_N, 32], 1.0e30, tl.float32)
        for j in tl.static_range(32):
            gj = GRID[j]
            lt = mag > gj  # grid[j] < mag
            idx += lt.to(tl.int32)
            glo = tl.where(lt, tl.maximum(glo, gj), glo)
            ge = mag <= gj  # grid[j] >= mag
            ghi = tl.where(ge, tl.minimum(ghi, gj), ghi)
        lo = tl.maximum(idx - 1, 0)
        dlo = mag - glo
        dhi = ghi - mag
        pick_hi = (dhi < dlo) | ((dhi == dlo) & ((lo & 1) == 1))
        chosen = tl.where(pick_hi, idx, lo)
        chosen = tl.minimum(tl.maximum(chosen, 0), 31)
        ybits = y.to(tl.int32, bitcast=True)
        sign = (ybits < 0).to(tl.int32) * 32
        codes = chosen | sign  # [BN,32] field-order 6-bit codes

        cf = codes.reshape(BLOCK_N, 8, 4)
        w = (1 << (6 * tl.arange(0, 4))).to(tl.int32)  # [1,6,12,18] shifts
        u = tl.sum(cf * w[None, None, :], axis=2)  # [BN,8] 24-bit packed words
        b0 = (u & 0xFF).to(tl.uint8)
        b1 = ((u >> 8) & 0xFF).to(tl.uint8)
        b2 = ((u >> 16) & 0xFF).to(tl.uint8)

        base = (bh * nT + t) * 12800  # tile byte base
        data_off = base + bn * 1536 + L * 24  # [BN]
        g = tl.arange(0, 8)
        off0 = data_off[:, None] + g[None, :] * 3
        tl.store(out_ptr + off0 + 0, b0, mask=m[:, None])
        tl.store(out_ptr + off0 + 1, b1, mask=m[:, None])
        tl.store(out_ptr + off0 + 2, b2, mask=m[:, None])
        # scale byte (d-major: 12288 + d_row*4 + kvblk)
        scale_off = base + 12288 + d_row * 4 + kvblk
        sb = ((E + 127) & 0xFF).to(tl.uint8)
        tl.store(out_ptr + scale_off, sb, mask=m)


# Single proven V layout: the historic "noswap" Triton name is kept as an alias.
quantize_fp6_v_noswap_triton = quantize_fp6_v_clean_triton


# ---------------------------------------------------------------------------
# Triton GPU QK packer (lastdim MXFP6-E2M3, eliminates the host QK pack)
# ---------------------------------------------------------------------------
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
        GRID: tl.constexpr,
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
        mag = tl.minimum(tl.abs(y), 7.5)

        idx = tl.zeros([BLOCK_N, 32], tl.int32)
        glo = tl.full([BLOCK_N, 32], -1.0e30, tl.float32)
        ghi = tl.full([BLOCK_N, 32], 1.0e30, tl.float32)
        for j in tl.static_range(32):
            gj = GRID[j]
            lt = mag > gj
            idx += lt.to(tl.int32)
            glo = tl.where(lt, tl.maximum(glo, gj), glo)
            ge = mag <= gj
            ghi = tl.where(ge, tl.minimum(ghi, gj), ghi)
        lo = tl.maximum(idx - 1, 0)
        dlo = mag - glo
        dhi = ghi - mag
        pick_hi = (dhi < dlo) | ((dhi == dlo) & ((lo & 1) == 1))
        chosen = tl.where(pick_hi, idx, lo)
        chosen = tl.minimum(tl.maximum(chosen, 0), 31)
        ybits = y.to(tl.int32, bitcast=True)
        sign = (ybits < 0).to(tl.int32) * 32
        codes = chosen | sign  # [BN,32] field-order codes

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
        KHS,  # nt*16384 output bytes per (b,h)
        SKH96,  # sk*h*96 = packed bytes per batch
        H,  # heads
        BLOCK: tl.constexpr,
    ):
        # One fused pass replacing the torch permute+contiguous / advanced-index gather /
        # masked_fill / buffer-copy chain (~4 full-size passes -> 1 gathered read + 1 write).
        pid = tl.program_id(0)
        nchunk = KHS // BLOCK
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
        tl.store(buf_ptr + bh * KHS + p, byte)


_QK_FIELD_PERM_CACHE: dict = {}


def _qk_field_perm_dev(device):
    """Cached per-device int32 field permutation for the lastdim fp6 pack. _qk_field_perm() is a
    compile-time constant, but rebuilding it + a PAGEABLE host->device copy on EVERY Q and K pack
    (quantize_fp6_k_lds_order_triton also calls the lastdim packer) was a per-attention sync that
    serialized the quant. Build once per device and reuse."""
    cperm = _QK_FIELD_PERM_CACHE.get(device)
    if cperm is None:
        cperm = torch.from_numpy(_qk_field_perm()).to(device)
        _QK_FIELD_PERM_CACHE[device] = cperm
    return cperm


def quantize_fp6_lastdim_triton(x: "torch.Tensor"):
    """GPU (Triton) equivalent of quantize_fp6_lastdim.

    x : torch float tensor [..., D] (D % 32 == 0) on GPU.
    Returns (packed uint8 [..., (D//32)*24], scale uint8 [..., D//32]) on the same
    device. Byte-identical to the numpy packer for inputs whose scaled values are
    exactly representable (e.g. bf16/fp16 Q/K, where v/2^E is an exponent shift);
    arbitrary fp32 inputs may differ by at most one code on measure-zero ties,
    which is within fp6 quantization noise."""
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
    BLOCK_N = 128
    grid = (triton.cdiv(n_blocks, BLOCK_N),)
    _pack_qk_fp6_kernel[grid](
        xflat,
        packed,
        scale,
        cperm,
        D,
        NB,
        n_blocks,
        GRID=_E2M3_GRID,
        BLOCK_N=BLOCK_N,
    )
    return (
        packed.reshape(*lead, NB * 24),
        scale.reshape(*lead, NB),
    )


# ---------------------------------------------------------------------------
# Kernel-ready packed views (GPU): bench / integration entry points
# ---------------------------------------------------------------------------
# These return tensors in the EXACT shape+stride the fwd_hd128_mxfp6 kernel
# consumes, so a consumer just hands them to flash_attn_mxfp4_func. They own the
# kernel-ABI knowledge -- the coalesced LDS-order K gather and the d-major
# tile-flat V byte strides -- that used to live in the benchmark. Both support
# S % 128 != 0 (the kernel masks the partial tail tile in softmax).

_K_LDS_GIDX_CACHE: dict = {}


def _k_lds_gather_index(nt: int, total: int, device):
    """Cached [(nt*16384)] LDS-order gather index + valid mask. valid = (g < total)
    zeroes BOTH the fp6 dup/overflow LDS tail AND a partial seq tail. Keyed by
    (nt, total, device): with a partial tail the same nt pairs with different total."""
    key = (nt, total, device)
    g = _K_LDS_GIDX_CACHE.get(key)
    if g is None:
        idx16k = torch.as_tensor(
            _k_lds_order_gather_index(), dtype=torch.long, device=device
        )
        full = (
            torch.arange(nt, device=device, dtype=torch.long) * 12288
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


def quantize_fp6_k_lds_order_triton(k_thd: "torch.Tensor", tile: int = 128, return_raw: bool = False):
    """GPU equivalent of quantize_fp6_k_lds_order, returning the kernel-ready K view
    (seq stride 128) + E8M0 scale. Byte-identical to the numpy oracle's buffer but
    runs the pack+gather entirely on-device (seconds vs the per-tile numpy CPU
    round-trip). Supports S % tile != 0 (the gather's valid mask zeroes the partial
    tail tile, which the kernel masks in softmax).

    k_thd : float K [b, sk, h, 128] on GPU.
    Returns (k_view uint8 [b, sk, h, 96] strided over a [b,h,n_tiles*16384] buffer,
             scale uint8 [b, sk, h, 4]).
    If return_raw: returns (buf, sbuf) -- the FULL contiguous backing buffers (uint8 1D)
    instead of the strided/padded views. A torch.library.custom_op caller MUST take this
    path: returning the strided k_view as a custom-op output lets AOTAutograd clone it to a
    contiguous numel-sized tensor (dropping the seq-stride-128 LDS layout -> garbage). The
    caller rebuilds k_view = buf.as_strided(...) and k_scale = sbuf[:n].view(...) OUTSIDE the
    op (traced as_strided, which inductor handles correctly)."""
    assert _HAVE_TRITON, "triton/torch unavailable"
    b, sk, h, d = k_thd.shape
    assert d == 128 and tile == 128, (d, sk, tile)
    nt = (sk + tile - 1) // tile  # ceil; partial tail handled by the valid mask
    packed, scale = quantize_fp6_lastdim_triton(k_thd)  # [b,sk,h,96], [b,sk,h,4]
    total = sk * 96
    # Fused on-device LDS reorder: a single Triton gather (read `packed` via the cached
    # source offset, apply the valid mask, write the buffer) replaces the torch chain of
    # permute+contiguous / advanced-index gather / masked_fill / buffer-copy (~4 full-size
    # passes -> 1 gathered read + 1 write; ~1.35x faster on the K reorder at long seq).
    srcw, valid8 = _k_lds_src_within(nt, total, h, k_thd.device)
    k_hs = nt * 16384
    k_bs = h * k_hs
    buf = torch.empty(b * k_bs + 256, dtype=torch.uint8, device=k_thd.device)
    BLOCK = 1024
    assert k_hs % BLOCK == 0, (k_hs, BLOCK)
    grid = (b * h * (k_hs // BLOCK),)
    _gather_k_lds_kernel[grid](
        packed.reshape(-1),
        buf,
        srcw,
        valid8,
        k_hs,
        sk * h * 96,
        h,
        BLOCK=BLOCK,
        num_warps=4,
    )
    # seq stride = tile (=128) -> the kernel's _s_k_Seqs=128 -> tile base = token*128.
    k_view = buf.as_strided((b, sk, h, 96), (k_bs, tile, k_hs, 1))
    # K-SCALE TRAILING PAD (gfx950 fwd_hd128_mxfp6 op_sel-shift-free K-scale path): the kernel's
    # coalesced K-scale gather keeps a 2nd "Region B" pre-shifted LDS copy, gathered by reading each
    # token's 4-byte E8M0 dword at a +1 BYTE source offset (so MFMA op_sel byte0/byte2 land dblk1/
    # dblk3 directly, no runtime >>blk1*8 shift). On the final (token,head,batch) that +1 dword read
    # runs 1 byte past the contiguous scale tensor -> re-home `scale` into a buffer with trailing
    # slack so the read stays mapped. The pad byte is never consumed by the MFMA (op_sel ignores it).
    sflat = scale.reshape(-1)
    sbuf = torch.empty(sflat.numel() + 64, dtype=torch.uint8, device=scale.device)
    sbuf[: sflat.numel()] = sflat
    if return_raw:
        return buf, sbuf
    scale = sbuf[: sflat.numel()].view(b, sk, h, 4)
    return k_view, scale


def pack_fp6_v_kernel_view(
    v_fp8: "torch.Tensor", tile: int = 128, use_triton: bool = True, out_device=None
):
    """Pack raw fp8 V into the kernel's native fp6 d-major tile-flat HBM layout and
    return it as a [b, sk, h_kv, d] view with the kernel's byte strides
    (v_Seqs=100, v_Hs=n_tiles*12800, v_Bs=h_kv*v_Hs). The per-channel v_descale is
    applied in the kernel epilogue, so this is a layout cast only. Supports
    S % tile != 0 by EDGE-padding the partial tail tile (replicate the last token so
    every E8M0 32-block keeps a finite magnitude -- a zero block could dequant
    0*inf -> NaN; the kernel masks tokens >= sk, so the padding never reaches out).

    v_fp8 : torch fp8 V [b, sk, h_kv, d=128]. use_triton=False forces the numpy
    host pack. out_device: move the final buffer here (the numpy pack lands on CPU).
    Returns uint8 view [b, sk, h_kv, d]."""
    assert _HAVE_TRITON, "triton/torch unavailable"
    b, sk, h_kv, d = v_fp8.shape
    n_tiles = (sk + tile - 1) // tile
    sk_pad = n_tiles * tile
    if sk_pad != sk:
        tail = v_fp8[:, sk - 1 : sk].expand(b, sk_pad - sk, h_kv, d)
        v_in = torch.cat([v_fp8, tail], dim=1)
    else:
        v_in = v_fp8
    if use_triton and _HAVE_TRITON:
        packed_flat = quantize_fp6_v_clean_triton(v_in, tile=tile).reshape(-1)
    else:
        v_f = v_in.detach().to(torch.float32).cpu().numpy()  # [b, sk_pad, h_kv, d]
        v_dmajor = np.transpose(v_f, (0, 2, 3, 1))  # [b, h_kv, d, sk_pad]
        packed = quantize_fp6_v_clean(v_dmajor, tile=tile)
        packed_flat = torch.from_numpy(np.ascontiguousarray(packed).reshape(-1))
    tile_bytes = d * 96 + d * 4  # 12800 for d=128
    v_hs = n_tiles * tile_bytes
    v_bs = h_kv * v_hs
    # as_strided can read up to (sk-1)*100 + (h_kv-1)*v_hs + (d-1), slightly past
    # b*v_bs; the +256 tail keeps the view in-bounds.
    buf = torch.empty(b * v_bs + 256, dtype=torch.uint8, device=packed_flat.device)
    buf[: packed_flat.numel()] = packed_flat
    if out_device is not None:
        buf = buf.to(out_device)
    return buf.as_strided((b, sk, h_kv, d), (v_bs, 100, v_hs, 1))


