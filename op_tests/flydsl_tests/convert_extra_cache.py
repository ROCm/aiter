# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Convert a packed fp8_ds_mla region1 cache from OCP e4m3 to fnuz e4m3 in place.

Region1's NoPE bytes are stored OCP, so the prefill kernel has to run
``_load_nope_convert`` on every region1 tile: dequantize, apply the UE8M0
per-64-block exponent, requantize to fnuz, ``ds_write``. Region0 instead stores
fnuz already and gets the fire-and-forget ``buffer_load_to_lds`` DMA path, which
is why region1 tiles cost ~1.8x region0 tiles.

The conversion is *not* an approximation: ``_load_nope_convert`` already ends in
``cvt_pk_fp8_f32``, so the bytes the MFMA consumes today are exactly fnuz e4m3.
Doing that once up front and letting region1 take the DMA path
(``r1_convert=False, extra_is_fnuz=True``) is bit-identical and measured -6.25%.

Only the 448 NoPE bytes of each 576-byte token record are touched; the bf16 RoPE
tail and the per-block scale trailer are left alone. With unity UE8M0 scales the
result is exact; with non-unity scales the per-block exponent is folded in, which
is the same fold the kernel does inline.

Range caveat: OCP e4m3 reaches 448 while fnuz e4m3 stops at 240, so the 14 byte
values encoding |v| in 256..448 are not representable and this kernel saturates
them to max-finite (Torch's cast instead yields 0x80, which is fnuz NaN). Such
values already violate the format contract the kernel assumes -- region0 stores
fnuz natively and hits the same ceiling -- and do not occur in practice, which is
why the end-to-end result is bit-identical on real caches.

Run:
    python op_tests/flydsl_tests/convert_extra_cache.py        # self-test + timing
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

NOPE_BYTES = 448  # 7 x 64-element blocks of fp8
TOKEN_BYTES = 576  # 448 NoPE + 64 RoPE bf16 (128 B)
CACHE_ROW = 584  # TOKEN_BYTES + 8 scale bytes


@triton.jit
def _convert_kernel(
    u8_ptr,
    i16_ptr,
    n_tokens,
    BS: tl.constexpr,
    DO_NOPE: tl.constexpr,
    DO_ROPE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    NOPE: tl.constexpr,
    ROPE: tl.constexpr,
    ROW: tl.constexpr,
    TOK: tl.constexpr,
    TOKS: tl.constexpr,
    NBLK: tl.constexpr,
    EXACT: tl.constexpr,
):
    """Both conversions in one pass over a token record.

    A token's NoPE bytes and its RoPE tail are adjacent, so doing them together
    reads the record once instead of twice. ``TOKS`` tokens per program is what
    makes this bandwidth-bound rather than launch-bound: one token is 448 B of NoPE
    or 128 B of RoPE, far too little to fill a wave.
    """
    pid = tl.program_id(0)
    t = (pid * TOKS + tl.arange(0, TOKS)).to(tl.int32)
    # Cache sizes are block-aligned, so the tail mask is normally dead weight on
    # every access; EXACT lets it fold away.
    tm = tl.full((TOKS,), 1, tl.int1) if EXACT else t < n_tokens
    # BS is constexpr and a power of two, so these are a shift and a mask rather
    # than the integer divide a runtime block_size would cost per token.
    blk = t // BS
    pos = t % BS
    base = blk * (BS * ROW) + pos * TOK

    if DO_NOPE:
        off = tl.arange(0, NBLK).to(tl.int32)
        m = tm[:, None] & (off[None, :] < NOPE)
        a = base[:, None] + off[None, :]
        v = tl.load(u8_ptr + a, mask=m, other=0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        v = tl.where(v != v, 0.0, v)
        if HAS_SCALE:
            enc = tl.load(
                u8_ptr
                + (blk * (BS * ROW) + BS * TOK + pos * 8)[:, None]
                + (off // 64)[None, :],
                mask=m,
                other=127,
            ).to(tl.float32)
            v = v * tl.exp2(enc - 127.0)
        tl.store(u8_ptr + a, v.to(tl.float8e4b8).to(tl.uint8, bitcast=True), mask=m)

    if DO_ROPE:
        ro = tl.arange(0, ROPE).to(tl.int32)
        rm = tm[:, None]
        # bf16 source is 2 B per element, so index the i16 view; the fp8 result needs
        # only ROPE bytes and lands at the front of the 2*ROPE the bf16 occupied,
        # which is exactly where _load_nope_dma's block-7 addressing looks.
        bits = tl.load(i16_ptr + ((base + NOPE) // 2)[:, None] + ro[None, :], mask=rm)
        rv = bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
        rv = tl.where(rv != rv, 0.0, rv)
        tl.store(
            u8_ptr + (base + NOPE)[:, None] + ro[None, :],
            rv.to(tl.float8e4b8).to(tl.uint8, bitcast=True),
            mask=rm,
        )


def convert_cache_(
    cache: torch.Tensor, *, nope: bool, rope: bool, has_scale: bool = False,
    toks: int = 8, num_warps: int = 4,
) -> torch.Tensor:
    """In-place cache format conversion, one pass.

    ``nope`` runs OCP -> fnuz on the 448 NoPE bytes (region1 only; region0 already
    stores fnuz). ``rope`` runs bf16 -> fp8 on the tail. Doing both in one call
    reads each token record once.
    """
    assert cache.dtype == torch.uint8 and cache.ndim == 3
    assert cache.shape[2] == CACHE_ROW, f"expected row {CACHE_ROW}, got {cache.shape[2]}"
    if not (nope or rope):
        return cache
    num_blocks, block_size, _ = cache.shape
    n_tokens = num_blocks * block_size
    flat = cache.view(-1)
    _convert_kernel[(triton.cdiv(n_tokens, toks),)](
        flat,
        flat.view(torch.int16),
        n_tokens,
        BS=block_size,
        DO_NOPE=nope,
        DO_ROPE=rope,
        HAS_SCALE=has_scale,
        NOPE=NOPE_BYTES,
        ROPE=64,
        ROW=CACHE_ROW,
        TOK=TOKEN_BYTES,
        TOKS=toks,
        NBLK=512,
        EXACT=(n_tokens % toks == 0),
        num_warps=num_warps,
    )
    return cache


def convert_extra_cache_(cache: torch.Tensor, *, has_scale: bool = False) -> torch.Tensor:
    """OCP -> fnuz on the NoPE bytes only. Prefer ``convert_cache_`` when the RoPE
    tail also needs quantizing, so the record is read once rather than twice."""
    return convert_cache_(cache, nope=True, rope=False, has_scale=has_scale)


def quantize_rope_(cache: torch.Tensor) -> torch.Tensor:
    """bf16 -> fnuz fp8 on the RoPE tail only.

    Destructive: the bf16 rope is gone afterwards, so a cache converted this way
    cannot be used with ``rope_bf16=True``, which needs it for the QK dot.
    """
    return convert_cache_(cache, nope=False, rope=True)


def _rope_reference(cache: torch.Tensor) -> torch.Tensor:
    """Torch equivalent of the RoPE tail quantization, used only to check the kernel."""
    out = cache.clone()
    nb, bs, _ = cache.shape
    flat = out.view(-1)[: nb * bs * CACHE_ROW].view(nb, bs * CACHE_ROW)
    tok = flat[:, : bs * TOKEN_BYTES].view(nb, bs, TOKEN_BYTES)
    src = tok[:, :, NOPE_BYTES:].reshape(-1).view(torch.bfloat16).to(torch.float32)
    src = torch.where(torch.isnan(src), torch.zeros_like(src), src)
    q = src.to(torch.float8_e4m3fnuz).view(torch.uint8).reshape(nb, bs, 64)
    tok[:, :, NOPE_BYTES : NOPE_BYTES + 64].copy_(q)
    return out


def _reference(cache: torch.Tensor, has_scale: bool = False) -> torch.Tensor:
    """Torch equivalent, used only to check the kernel."""
    out = cache.clone()
    nb, bs, _ = cache.shape
    flat = out.view(-1)[: nb * bs * CACHE_ROW].view(nb, bs * CACHE_ROW)
    nope = flat[:, : bs * TOKEN_BYTES].view(nb, bs, TOKEN_BYTES)[:, :, :NOPE_BYTES]
    f = nope.reshape(-1).view(torch.float8_e4m3fn).to(torch.float32)
    f = torch.where(torch.isnan(f), torch.zeros_like(f), f)
    f = f.reshape(nb, bs, NOPE_BYTES)
    if has_scale:
        # UE8M0 trailer: 8 exponent bytes per token, one per 64-element block,
        # living after the block's token records. Only the first 7 cover NoPE's
        # 448 bytes; the 8th belongs to the RoPE block.
        enc = flat[:, bs * TOKEN_BYTES :].view(nb, bs, 8)[:, :, : NOPE_BYTES // 64]
        f = f * torch.exp2(enc.to(torch.float32) - 127.0).repeat_interleave(64, dim=2)
    nope.copy_(f.to(torch.float8_e4m3fnuz).view(torch.uint8).reshape(nope.shape))
    return out


def main() -> int:
    import os
    import sys
    import time

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from sparse_mla_prefill_ref import gen_kv, pack_fp8_ds_mla_cache

    torch.manual_seed(0)
    bs = 64
    # Realistic contents: the same packer the benchmark and tests use. Uniform
    # random bytes would instead sweep OCP's unrepresentable 256..448 range and
    # its NaN encodings, which no real cache contains.
    cache = pack_fp8_ds_mla_cache(gen_kv(512 * bs, seed=0), bs, is_extra=True)
    nb = cache.shape[0]

    ref = _reference(cache)
    got = convert_extra_cache_(cache.clone())
    same = torch.equal(ref, got)
    bad = (ref != got).sum().item()
    print(f"  nope OCP->fnuz vs torch: bit-identical={same}  mismatched bytes={bad}")

    # has_scale reads the UE8M0 trailer, whose per-block stride is only exercised
    # once more than one block is populated with non-127 exponents.
    scaled = cache.clone()
    scaled.view(-1)[: nb * bs * CACHE_ROW].view(nb, bs * CACHE_ROW)[:, bs * TOKEN_BYTES :] = (
        torch.randint(120, 132, (nb, bs * 8), dtype=torch.uint8, device=cache.device)
    )
    sref = _reference(scaled, has_scale=True)
    sgot = convert_extra_cache_(scaled.clone(), has_scale=True)
    ssame = torch.equal(sref, sgot)
    print(f"  nope w/ UE8M0 scale vs torch: bit-identical={ssame}  "
          f"mismatched bytes={(sref != sgot).sum().item()}")
    same = same and ssame

    rref = _rope_reference(cache)
    rgot = quantize_rope_(cache.clone())
    rsame = torch.equal(rref, rgot)
    rbad = (rref != rgot).sum().item()
    print(f"  rope bf16->fp8 vs torch: bit-identical={rsame}  mismatched bytes={rbad}")
    same = same and rsame

    mb = cache.numel() / 2**20
    for label, fn in (("triton", lambda c: convert_extra_cache_(c)),
                      ("torch ", lambda c: _reference(c))):
        work = cache.clone()
        for _ in range(3):
            fn(work)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(20):
            fn(work)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t) / 20 * 1000
        print(f"  {label} conversion: {ms:.4f} ms for {mb:.1f} MB "
              f"({2 * mb * NOPE_BYTES / CACHE_ROW / ms / 1024:.2f} GB/s effective)")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
