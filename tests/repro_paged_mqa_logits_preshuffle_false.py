#!/usr/bin/env python3
"""Repro for ROCm/aiter `deepgemm_fp8_paged_mqa_logits` Preshuffle=False bugs.

Run against a ROCm/aiter main checkout (unpatched upstream):

    PYTHONPATH=/path/to/ROCm/aiter python3 repro_paged_mqa_logits.py

Expected results on unpatched upstream (gfx950):
  issue 1 -> RuntimeError: Triton Error [HIP]: Code: 700
             (illegal memory access; kv_indices read per token position runs
              off the block-level page table)
  issue 2 -> allclose=False with thousands of polluted scores
             (the same out-of-contract reads land in valid-but-wrong memory)

Root cause: the Preshuffle=False kernel is built for per-token paging
(KVBlockSize=1, 132B/token interleaved). With KVBlockSize=64 + block-level
page tables it (a) reads kv_indices per token position -> OOB, and
(b) forms KV addresses without the in-block offset -> only token 0 of each
64-token block is reachable.

Only torch + the official aiter package are required.
"""

import torch
from aiter.ops.triton.attention.pa_mqa_logits import (
    deepgemm_fp8_paged_mqa_logits)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FP8 = torch.float8_e4m3fn
BLOCK_SIZE = 64


def make_q(batch, next_n, heads, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return torch.randn(batch, next_n, heads, 128, generator=g, device=DEVICE,
                       dtype=torch.bfloat16).to(FP8)


def make_sectioned_cache(num_blocks, seed=1):
    """[num_blocks, 64, 1, 132] uint8, per block: 64x128B fp8 values | 64x4B
    fp32 scales (the vLLM DSA indexer cache layout)."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    kv = torch.randn(num_blocks * BLOCK_SIZE, 128, generator=g, device=DEVICE,
                     dtype=torch.bfloat16)
    amax = kv.abs().amax(dim=-1, keepdim=True).clamp(1e-4)
    sf = amax / torch.finfo(FP8).max
    kv_q = (kv / sf).to(FP8).view(num_blocks, BLOCK_SIZE, 128)
    kv_sf = sf.squeeze(-1).float().view(num_blocks, BLOCK_SIZE)
    cache = torch.empty(num_blocks, BLOCK_SIZE, 1, 132, dtype=torch.uint8,
                        device=DEVICE)
    flat = cache.view(num_blocks, BLOCK_SIZE * 132)
    flat[:, :BLOCK_SIZE * 128] = kv_q.view(num_blocks, -1).view(torch.uint8)
    flat[:, BLOCK_SIZE * 128:] = kv_sf.view(num_blocks, -1).view(torch.uint8)
    return cache, kv_q, kv_sf


def issue1_illegal_address():
    """Block-level page table [B, 16] + KVBlockSize=64."""
    batch, next_n, heads, ctx = 2, 1, 32, 997
    nb = (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE  # 16 blocks
    q = make_q(batch, next_n, heads)
    cache, _, _ = make_sectioned_cache(batch * nb)
    g = torch.Generator(device=DEVICE).manual_seed(3)
    weights = torch.randn(batch * next_n, heads, generator=g, device=DEVICE,
                          dtype=torch.float32)
    context_lens = torch.full((batch, ), ctx, dtype=torch.int32, device=DEVICE)
    block_tables = torch.arange(batch * nb, dtype=torch.int32,
                                device=DEVICE).view(batch, nb)
    out = torch.full((batch * next_n, ctx),
                     float("-inf"),
                     dtype=torch.float32,
                     device=DEVICE)
    deepgemm_fp8_paged_mqa_logits(q, cache, weights, out, context_lens,
                                  block_tables, ctx,
                                  Preshuffle=False,
                                  KVBlockSize=BLOCK_SIZE,
                                  ChunkK=256,
                                  WavePerEU=2)
    torch.cuda.synchronize()
    print("[issue1] NO CRASH (kernel did not fault -- unexpected on unpatched "
          "upstream)")


def issue2_score_pollution():
    """Large padded table: OOB reads land in valid-but-wrong memory."""
    batch, next_n, heads, ctx = 2, 1, 32, 3000
    nb = (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE  # 47 blocks
    q = make_q(batch, next_n, heads)
    cache, kv_q, kv_sf = make_sectioned_cache(nb)
    g = torch.Generator(device=DEVICE).manual_seed(4)
    weights = torch.randn(batch * next_n, heads, generator=g, device=DEVICE,
                          dtype=torch.float32)
    context_lens = torch.full((batch, ), ctx, dtype=torch.int32, device=DEVICE)
    max_bl = 4096
    block_tables = torch.zeros(batch, max_bl, dtype=torch.int32, device=DEVICE)
    perm = torch.randperm(nb, generator=g, device=DEVICE).to(torch.int32)
    block_tables[0, :nb] = perm
    block_tables[1, :nb] = perm.flip(0)

    out = torch.full((batch * next_n, ctx),
                     float("-inf"),
                     dtype=torch.float32,
                     device=DEVICE)
    deepgemm_fp8_paged_mqa_logits(q, cache, weights, out, context_lens,
                                  block_tables, ctx,
                                  Preshuffle=False,
                                  KVBlockSize=BLOCK_SIZE,
                                  ChunkK=256,
                                  WavePerEU=2)
    torch.cuda.synchronize()

    # fp32 reference with correct block addressing
    ref = torch.full((batch * next_n, ctx),
                     float("-inf"),
                     dtype=torch.float32,
                     device=DEVICE)
    for b in range(batch):
        idx = block_tables[b, :nb].long()
        k = (kv_q[idx].float() * kv_sf[idx][..., None]).view(-1, 128)[:ctx]
        s = (q[b, 0].float() @ k.T).relu()
        ref[b, :ctx] = (s * weights[b][:, None]).sum(0)

    valid = torch.isfinite(ref)
    got, exp = out[valid], ref[valid]
    ok = torch.allclose(got, exp, rtol=1e-2, atol=1e-2)
    mism = int((~torch.isclose(got, exp, rtol=1e-2, atol=1e-2)).sum())
    print(f"[issue2] allclose={ok} mismatch={mism}/{int(valid.sum())} "
          f"(unpatched upstream: allclose=False with thousands mismatched)")


if __name__ == "__main__":
    assert DEVICE == "cuda", "requires a ROCm GPU"
    print("gfx:", torch.cuda.get_device_properties(0).gcnArchName)
    issue2_score_pollution()
    # issue1 last: on unpatched upstream this faults with HIP error 700 and
    # may abort the process -- that abort IS the reproduction.
    issue1_illegal_address()
