# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness test for the dedicated CK-free HIP KV-cache write module
(module_cache_reshape): aiter.ops.cache.reshape_and_cache vs a torch reference,
for the HND and 5D-SHUFFLE (asm) paged layouts, bf16 + fp16, with -1 padding.
Runs the HIP kernel on the live GPU (incl. gfx1201 RDNA4)."""

import torch
import aiter
import aiter.ops.cache as cache


def ref_scatter(key, value, slot, nb, KH, D, block_size, x, asm):
    # reference on CPU; index math identical to reshape_and_cache_kernel
    key, value, slot = key.cpu(), value.cpu(), slot.cpu()
    dt = key.dtype
    kc = torch.zeros((nb, KH, D // x, block_size, x), dtype=dt)
    if asm:
        vc = torch.zeros((nb, KH, block_size // x, D, x), dtype=dt)
    else:
        vc = torch.zeros((nb, KH, D, block_size), dtype=dt)
    N = key.shape[0]
    for t in range(N):
        sl = int(slot[t].item())
        if sl < 0:
            continue
        b, w = sl // block_size, sl % block_size
        for h in range(KH):
            for d in range(D):
                kc[b, h, d // x, w, d % x] = key[t, h, d]
                if asm:
                    vc[b, h, w // x, d, w % x] = value[t, h, d]
                else:
                    vc[b, h, d, w] = value[t, h, d]
    return kc, vc


def run_case(dtype, asm):
    dev = "cuda"
    torch.manual_seed(0)
    N, KH, D, block_size, nb = 20, 8, 128, 64, 4
    x = 16 // torch.empty(0, dtype=dtype).element_size()
    key = torch.randn(N, KH, D, device=dev, dtype=dtype)
    value = torch.randn(N, KH, D, device=dev, dtype=dtype)
    slot = torch.arange(N, device=dev, dtype=torch.int64)
    slot[3] = -1
    slot[7] = -1  # padded tokens must be skipped
    kc = torch.zeros((nb, KH, D // x, block_size, x), device=dev, dtype=dtype)
    if asm:
        vc = torch.zeros((nb, KH, block_size // x, D, x), device=dev, dtype=dtype)
    else:
        vc = torch.zeros((nb, KH, D, block_size), device=dev, dtype=dtype)
    cache.reshape_and_cache(key, value, kc, vc, slot, "auto", None, None, asm)
    torch.cuda.synchronize()
    rkc, rvc = ref_scatter(key, value, slot, nb, KH, D, block_size, x, asm)
    okk = torch.equal(kc.cpu(), rkc)
    okv = torch.equal(vc.cpu(), rvc)
    tag = "SHUFFLE/asm" if asm else "HND"
    print(f"[{tag:11s} {str(dtype):14s}] K match={okk}  V match={okv}")
    assert okk and okv, f"mismatch: {tag} {dtype}"


if __name__ == "__main__":
    for asm in (True, False):
        for dtype in (torch.bfloat16, torch.float16):
            run_case(dtype, asm)
    print("ALL_PASS")
