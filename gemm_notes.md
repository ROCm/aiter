# CK Tiny vs T-2D GEMM Shape Correspondence

Workload: `aiter_unified_attention_foo.jsonl` — d64, GQA-8, `block_size=32`, SWA `[127,0]`, decode, bf16.

## Kernel selection

For this workload, CK routes to `unified_attention_decode_tiny_kernel_traits`, specifically the **bs32 + local** instance:

```cpp
// unified_attention.cpp
#define DISPATCH_UNIFIED_ATTENTION_DECODE_TINY_BS32_LOCAL(DType, HSize, BM, NQPKV) \
    { \
        using kernel_traits = unified_attention_decode_tiny_kernel_traits<DType, /*IsMasking=*/true, HSize, BM, NQPKV, 32, /*IsLocal=*/true>; \
        return unified_attention_kernel_dispatch_decode<kernel_traits>(args, config); \
    }
```

Instantiated traits:

```
unified_attention_decode_tiny_kernel_traits<bf16, true, 64, 16, 8, 32, true>
```

Not the template default `BlockSize_=64` — `page_blk_size=32` forces the explicit `32` argument. The traits comment notes this tier was designed to match Triton decode:

```cpp
// unified_attention_impl.hpp
// Tiny decode traits: 1 warp, 16x16 MFMA, kBlockM=16, kBlockQ=2 for GQA-8.
// Matches Triton's BLOCK_M=16 / BLOCK_Q=2 decode configuration.
```

## Block-level tiling

CK `unified_attention_block_tile` is `sequence<kBlockM, kBlockQ, BLOCK_SIZE, HEAD_SIZE>`:

| CK trait field | Value | T-2D equivalent |
|---|---|---|
| `kBlockM` | 16 | `BLOCK_M` |
| `kBlockQ` | 2 | `BLOCK_Q` |
| `kPageBlockSize` / `BLOCK_SIZE` | 32 | `TILE_SIZE` (= `block_size`) |
| `kHeadDim` | 64 | `HEAD_SIZE` |
| `kHeadDimPadded` | 64 | `HEAD_SIZE_PADDED` |
| `num_queries_per_kv` | 8 | GQA-8 |

The outer attention tile is the same: one WG covers 16 Q-rows (8 live), walks KV in chunks of 32 along sequence, head dim 64.

**Grid** (aligned for this shape):

- CK decode: `(num_kv_heads, num_seqs)` = `(8, 1)`
- T-2D: `(num_kv_heads, total_num_q_blocks)` = `(8, 1)`

**SWA iteration**: 5 KV tiles (key positions 873–1000, window 128).

## GEMM shapes: CK splits QK and PV internally

Even though the traits struct uses one `unified_attention_warp_gemm_shape` for both, the pipeline distinguishes the two GEMMs via `GetQKBlockGemm` / `GetPVBlockGemm`.

**Gemm0 (QK)** — block tile `sequence<M, N, K>` = `sequence<16, 32, 64>`:

```
Q [16 × 64]  @  K [64 × 32]  →  S [16 × 32]
```

**Gemm1 (PV)** — block tile `sequence<M, N, K>` = `sequence<16, 64, 32>`:

```
P [16 × 32]  @  V [32 × 64]  →  O [16 × 64]
```

Identical to T-2D per-tile `tl.dot` shapes:

```python
# unified_attention.py (kernel)
S = qk_scale * tl.dot(Q, K)           # Q:(16,64), K:(64,32) transposed layout
acc += tl.dot(P.to(V.dtype), V)       # P:(16,32), V:(32,64)
```

At the block-GEMM level, correspondence is essentially 1:1. The traits do not expose two separate warp shapes, but the pipeline instantiates two different `TileGemmShape`s from the same block tile parameters.

## Where they diverge: warp/MFMA granularity

Warp-level MFMA tile from traits:

```cpp
// unified_attention_impl.hpp
using unified_attention_warp_gemm_shape = sequence<16, 16, 32>;
// 1 warp: kBlockM=1*16=16, kBlockSize=64, NumWarpGroups=1
using unified_attention_block_warps     = sequence<1, 1, 1>;
```

With 1 warp covering the full block, each QK block GEMM `16×32×64` decomposes into:

| Axis | Block size | Warp tile | Tiles per warp |
|---|---|---|---|
| M | 16 | 16 | 1 |
| N (KV seq) | 32 | 16 | 2 |
| K (head) | 64 | 32 | 2 |

→ **4 warp MFMAs** per QK block GEMM (and similarly for PV).

T-2D uses `tl.dot` without an explicit warp tile; on MI300 that typically lowers to **16×16×16** bf16 MFMAs, giving **8 MFMAs** for the same `16×64 @ 64×32` dot.

| Level | Correspondence |
|---|---|
| Logical attention per KV head | Same: `[8×64]@[64×128]` then `[8×128]@[128×64]` |
| Per-KV-tile block GEMM | Same: `16×32×64` QK, `16×64×32` PV |
| Warp/MFMA decomposition | Different: CK `16×16×32` × 1 warp vs Triton ~`16×16×16` |
| Warps / occupancy | Different: CK 1 warp (64 threads) vs Triton 2 warps |

## Conceptual mapping summary

```
                    T-2D                          CK tiny (bs32 local)
                    ─────                         ────────────────────
Grid                (8 kv_heads, 1 q_block)       (8 kv_heads, 1 seq)     ✓ same
Q tile              BLOCK_M=16, BLOCK_Q=2         kBlockM=16, kBlockQ=2   ✓ same
KV tile             TILE_SIZE=32                  kPageBlockSize=32       ✓ same
Head dim            64                            64                      ✓ same
QK block GEMM       [16×64]@[64×32]→[16×32]      Gemm0 16×32×64          ✓ same
PV block GEMM       [16×32]@[32×64]→[16×64]      Gemm1 16×64×32          ✓ same
SWA tiles/WG        5                             5 (IsLocal=true)        ✓ same
Warp GEMM atom      ~16×16×16 (implicit)          16×16×32 (explicit)     ✗ differs
Warps/block         2                             1                       ✗ differs
Pipeline            separate tl.dot calls         fused CK pipeline       structural diff
```

## Bottom line

`unified_attention_decode_tiny_kernel_traits` (with `BlockSize_=32`, `IsLocal_=true`) is the right CK counterpart to T-2D for this workload. Macroscopic shapes match closely — same block tile `<16, 2, 32, 64>`, same per-tile QK/PV GEMM dimensions, same grid topology, same SWA tile count.

The performance gap is unlikely to come from a shape mismatch at the block level. More likely causes:

1. **Warp/MFMA atom size** — `16×16×32` vs Triton's effective `16×16×16` tiling
2. **Occupancy** — 1 warp vs 2 warps
3. **Fused pipeline vs separate dots** — CK's `UnifiedAttentionPipelineTinyDecodePolicy` fuses softmax + PV; Triton does explicit `exp2`/softmax between dots
4. **Memory access patterns** — CK async-copy/LDS layout vs Triton direct loads with `.cg` cache modifier in decode

Conceptually the shapes are intentionally aligned — the CK tiny tier was built to mirror T-2D's decode tiling. The remaining ~3 µs kernel gap is more about **how** those same-shaped GEMMs are executed, not **what** shapes they use.
