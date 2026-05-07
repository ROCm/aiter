# aiter ASM paged-attention `block_id` truncation at the 2 GB / 65,536-block boundary

## Summary

When the `block_id` value loaded from the `block_tables` tensor reaches **65,536**, aiter's precompiled ASM `pa_*.co` paged-attention kernels on `gfx950` read from the wrong physical KV slot. The wrong-slot index matches `block_id & 0xFFFF`, consistent with a 16-bit narrowing of `block_id` somewhere in the slot-address path before the multiply by per-block stride.

Equivalently: the bug fires the moment `block_id × per_block_stride` reaches **2^31 bytes (2 GB)** — exactly the signed-i32 buffer-descriptor offset limit on AMD GPUs. For the production-relevant Eagle3 draft layout (`num_kv_heads=8, head_dim=128, block_size=16, bf16` → 32 KB per block), this happens at **block_id = 65,536**.

The bug is silent: no error, no NaN. Output collapses to whatever value lives at `block_id & 0xFFFF` (often zero in test, garbage in production), which propagates through softmax as a wrong attention result.

A self-contained microbench is included at `op_tests/test_pa_block_id_truncation.py`.

## Affected binaries

Reproduced on `gfx950` (AMD Instinct MI355X). Verified affected:

```
hsa/gfx950/pa/pa_bf16_noquant_gqa8_1tg_4w.co            (qlen=1, non-MTP)
hsa/gfx950/pa/pa_bf16_noquant_gqa8_1tg_4w_mtp_msk1.co   (qlen=4, MTP)
```

Both fail with the identical wrap signature, suggesting the truncation lives in shared ASM code rather than per-kernel logic. Likely the entire same-build `pa_*_blkSz=16` family (including the bf16/fp8 variants used in production).

C++ entry: `csrc/py_itfs_cu/asm_pa.cu:166 pa_fwd`.

## Reproduction

```bash
cd /root/aiter
HSA_NO_SCRATCH_RECLAIM=1 AITER_LOG_LEVEL=WARNING \
    python op_tests/test_pa_block_id_truncation.py
```

Or with pytest:

```bash
HSA_NO_SCRATCH_RECLAIM=1 \
    pytest op_tests/test_pa_block_id_truncation.py -v -s
```

The script allocates a 70,000-block KV pool (~4.6 GB total bf16 K+V), fingerprints four blocks with distinct constant values, then runs `pa_fwd_asm` once per block with a single-block sequence. Because each fingerprinted block is filled with a constant V, the attention output should equal that constant. If the kernel reads from a different block instead, the output collapses to whatever lives there (zero for unfilled blocks).

## Observed output

```
[OK]  block_id=   1000  expected=0.5000  actual=0.5000  Δ=+0.0000  (below_65535)
[OK]  block_id=  65535  expected=0.3000  actual=0.3008  Δ=+0.0008  (edge_65535_last_u16)
[BUG] block_id=  65536  expected=0.4000  actual=0.0000  Δ=-0.4000  (edge_65536_first_overflow)
  → if block_id is narrowed to 16 bits, reads block 0 instead (unfilled = 0).
[BUG] block_id=  67000  expected=0.7500  actual=0.0000  Δ=-0.7500  (above_65535)
  → if block_id is narrowed to 16 bits, reads block 1464 instead (unfilled = 0).
```

Both kernels (qlen=1 non-MTP and qlen=4 MTP) exhibit identical behavior.

| `block_id` | `block_id & 0xFFFF` | reads block | expected fingerprint | actual output | verdict |
|---:|---:|---:|---:|---:|---|
| 1,000 | 1,000 | 1,000 (= itself) | 0.5000 | 0.5000 | OK |
| 65,535 | 65,535 | 65,535 (= itself, no wrap) | 0.3000 | 0.3008 | OK |
| **65,536** | **0** | **0** (unfilled) | 0.4000 | **0.0000** | **BUG** |
| 67,000 | 1,464 | 1,464 (unfilled) | 0.7500 | 0.0000 | BUG |

Last safe value: `block_id = 65,535`. First failing value: `block_id = 65,536`.

## Root-cause hypothesis

For the test's per-block stride (Eagle3 draft layout):

```
slot_byte_offset = block_id × (block_size × num_kv_heads × head_dim × elem_size)
                 = block_id × (16 × 8 × 128 × 2)
                 = block_id × 32,768
```

| `block_id` | `slot_byte_offset` | meaning |
|---:|---:|---|
| 65,535 | 2,147,450,880 | 2 GB − 32 KB |
| **65,536** | **2,147,483,648** | **= 2^31 = 2 GB exactly = signed-i32 max + 1** |

AMD GPU buffer-descriptor offsets are signed i32, so a single descriptor cannot address more than ~2 GB. The kernel author likely reasoned:

> "Per-block stride is at most 32 KB, so `block_id` never exceeds 2 GB / 32 KB = 65,536. Store `block_id` as `u16` to save register space."

This holds when the entire KV cache fits in 2 GB. It breaks the moment any layer's KV pool exceeds 2 GB and `block_id` actually crosses 65,535 — exactly what happens for few-layer drafts (e.g. Eagle3) where the entire KV budget concentrates on one layer (~4.4 GB at `--gpu-memory-utilization 0.8`).

## Production impact

This bug was discovered during ATOM Eagle3 spec-decoding investigation on Kimi-K2.5 + kimi-k2.5-eagle3 (8× MI355X, TP=8, bf16 KV). Symptoms:

- Sharp, permanent acceptance-rate collapse from ~80% to ~10–25% the first time the draft KV pool's `block_id_amax` crosses 65,535.
- 100% reproducible across 7+ independent runs at `--gpu-memory-utilization 0.8`.
- `--gpu-memory-utilization 0.4` keeps `num_blocks < 30k` and the bug never fires (workaround).

ATOM's current workaround is to route pure-MHA layers (Eagle3 draft) through the triton/gluon paged-attention path instead of the aiter ASM family.

## Suggested fix

Audit ASM source for the affected `pa_*.co` kernels. For every load from the `block_tables` tensor, ensure:

1. `block_id` stays in i32 (or i64) all the way through the slot-address computation — never narrowed via `s_pack_*_b16` / `v_pack_*_b16` / `AND 0xFFFF` to a 16-bit register.
2. The product `block_id × per_block_stride` and the final add of the within-block offset are both 64-bit.
3. When the total KV cache exceeds 2 GB, span it across multiple buffer descriptors (chunked by `0x7FFF0000` byte windows) and switch descriptors as `block_id` crosses chunk boundaries — analogous to FlyDSL's `_chunk_buffer_resource_for_block`.

A regression test reproducing the bug is at `op_tests/test_pa_block_id_truncation.py`. Once the ASM is fixed, that test should report `[OK]` for all four block IDs.

## Environment

- Hardware: 8× AMD Instinct MI355X (`gfx950`)
- OS: Ubuntu 24.04.4 LTS, kernel 6.8.0-65-generic
- ROCm: 7.2.2 (HIP runtime 7.2.53211.70202-86~24.04, GPU firmware version 14)
- aiter: built locally from this branch
- Python: 3.12, PyTorch with ROCm support
