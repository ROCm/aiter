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
def _ocp_to_fnuz_kernel(
    cache_ptr,
    scale_ptr,
    n_tokens,
    block_size: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    NOPE: tl.constexpr,
    ROW: tl.constexpr,
    TOK: tl.constexpr,
    BLK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= n_tokens:
        return
    blk = pid // block_size
    pos = pid % block_size
    base = blk * (block_size * ROW) + pos * TOK

    offs = tl.arange(0, BLK)
    mask = offs < NOPE
    raw = tl.load(cache_ptr + base + offs, mask=mask, other=0)
    v = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    # The kernel's _flush_nan: NaN becomes 0 rather than propagating.
    v = tl.where(v != v, 0.0, v)
    if HAS_SCALE:
        # UE8M0 exponent byte per 64-element block. No OCP->fnuz term here: the
        # kernel's bias_f32=1.0 only compensates for gfx942's cvt_pk_f32_fp8 being
        # an *fnuz* instruction reading OCP bytes at half value, and float8e4nv
        # above already decodes with the OCP bias.
        sblk = offs // 64
        enc = tl.load(scale_ptr + blk * (block_size * 8) + pos * 8 + sblk,
                      mask=mask, other=127).to(tl.float32)
        v = v * tl.exp2(enc - 127.0)
    out = v.to(tl.float8e4b8).to(tl.uint8, bitcast=True)
    tl.store(cache_ptr + base + offs, out, mask=mask)


def convert_extra_cache_(cache: torch.Tensor, *, has_scale: bool = False) -> torch.Tensor:
    """In-place OCP -> fnuz on the NoPE bytes of a packed region1 cache.

    ``cache`` is ``[num_blocks, block_size, 584]`` uint8 as produced by
    ``pack_fp8_ds_mla_cache(..., is_extra=True)``. Pass ``has_scale=True`` when the
    UE8M0 trailer is not all 127, so the per-block exponent gets folded in.
    """
    assert cache.dtype == torch.uint8 and cache.ndim == 3
    assert cache.shape[2] == CACHE_ROW, f"expected row {CACHE_ROW}, got {cache.shape[2]}"
    num_blocks, block_size, _ = cache.shape
    n_tokens = num_blocks * block_size
    flat = cache.view(-1)
    # Scale trailer sits after the block's token records; same buffer, own offset.
    scale_view = flat[block_size * TOKEN_BYTES :] if has_scale else flat
    _ocp_to_fnuz_kernel[(n_tokens,)](
        flat,
        scale_view,
        n_tokens,
        block_size=block_size,
        HAS_SCALE=has_scale,
        NOPE=NOPE_BYTES,
        ROW=CACHE_ROW,
        TOK=TOKEN_BYTES,
        BLK=512,
        num_warps=4,
    )
    return cache


def _reference(cache: torch.Tensor) -> torch.Tensor:
    """Torch equivalent, used only to check the kernel."""
    out = cache.clone()
    nb, bs, _ = cache.shape
    flat = out.view(-1)[: nb * bs * CACHE_ROW].view(nb, bs * CACHE_ROW)
    nope = flat[:, : bs * TOKEN_BYTES].view(nb, bs, TOKEN_BYTES)[:, :, :NOPE_BYTES]
    f = nope.reshape(-1).view(torch.float8_e4m3fn).to(torch.float32)
    f = torch.where(torch.isnan(f), torch.zeros_like(f), f)
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
    print(f"  vs torch reference: bit-identical={same}  mismatched bytes={bad}")

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
