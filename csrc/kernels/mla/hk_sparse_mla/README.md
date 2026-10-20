# Sparse MLA forward (DSA) for gfx950

DeepSeek Sparse Attention's attention step: every query token attends to its own list of `topk` selected KV slots.
All heads share one latent KV row (MQA), so the same row is both K and V. Exposed as `aiter.hk_sparse_mla_fwd` (JIT
module `module_hk_sparse_mla`, ctypes entry points in `hk_sparse_mla_entry.cu`). The kernel implementation is
maintained here in AITER; HipKittens provides the tile and MMA primitives.

```
out[t][h][:] = softmax_k(q[t][h] . kv[idx[t][k]] * sm_scale) @ kv[idx[t][k]][:512]
```

## Contract

| | |
|---|---|
| `q` | `[T][32][512]` fp8 e4m3 (OCP), contiguous, no scales |
| `kv` | slot `s < num_slots` at `kv + s * kv_stride_bytes`; its first 512 bytes are the latent (fp8 e4m3, no scales). Rows and base must be 16-byte aligned (a 576-byte nope+rope row works as is). Slots outside `[0, num_slots)` are masked like `idx < 0` |
| `idx` | `[T][topk]` int32 slots; `idx < 0` is masked. `topk` must be a positive multiple of 128, up to 2304: pad with -1 |
| `out` | `[T][32][512]` bf16 |
| numerics | fp32 scores, P quantized as fp8(P * 448) before PV (same as the TileLang kernel); a row with no valid key outputs 0 |
| workspace | `hk_sparse_mla_workspace_bytes(T, topk)` bytes (0 when no split is used). The fp32 partials for the split-K path; allocate once for the largest decode batch, before graph capture |

```c
int    hk_sparse_mla_fwd(const void *q, const void *kv, const int *idx, void *out, int T, int topk,
                         long kv_stride_bytes, long num_slots, float sm_scale, void *workspace, size_t ws_bytes,
                         hipStream_t stream);
size_t hk_sparse_mla_workspace_bytes(int T, int topk);
int    hk_sparse_mla_splits(int T, int topk);   // key splits per token chosen for T
```

`hk_sparse_mla_fwd` returns 0, -1 (unsupported topk, misaligned kv or `num_slots` outside `[1, 2^31)`), -2 (launch failed) or -3 (workspace too small).
Any other head count or latent size is not supported, so callers fall back to another kernel.

## Design

- One 512-thread workgroup (8 waves, two per SIMD) per (token, key split), in 128-key blocks. Each wave scores 16
  keys of the block and accumulates 64 output dims. The second wave per SIMD hides part of the other's LDS and
  MFMA latency.
- The split's index list is staged in LDS once.
- A KV block is four HipKittens `st_fp8e4m3<128, 128>` shared tiles (st_16x128 swizzle), double-buffered. KV rows
  are gathered straight into them with `buffer_load ... lds`, where each lane's buffer offset is its slot times
  the row stride. Wave w fills the tile for dims 128w.
- Buffer offsets are 32-bit, so a pool of 4 GiB or more uses a second build of the kernel instead. It gathers with
  `global_load_lds` on 64-bit addresses and is about 6% slower at 16k tokens, with identical output.
- P goes through an `st_fp8e4m3<32, 128>` shared tile.
- Q, K, P and V are HipKittens fp8 register tiles (`rt_fp8e4m3`). K and P are read with HipKittens' row-layout
  loads, and V with its col-layout `ds_read_b64_tr_b8` load. S^T and O are fp32 accumulator tiles. Both matmuls
  are `mma_ABt_scaled` (`v_mfma_scale_f32_16x16x128_f8f6f4`) at unit scale, since the fp8 has no scales.
- Kernel-local:
  - the indexed gather;
  - the fp8 Q load, since HipKittens' global tile loads reject fp8 (Q uses the k order its shared-tile loads
    produce);
  - the bf16 output store, which rounds to nearest even where HipKittens' convert truncates;
  - the reductions.
- Scores are computed as `S^T = K Q^T`, so each lane holds 4 consecutive keys of one head. The softmax row max and sum
  then reduce with `permlane16/32_swap` instead of LDS round trips.
- `PV` reads V one 16-dim tile ahead of its MFMA.
- Small T (decode) splits each token's keys across workgroups until the grid covers the CUs. Each split writes an
  output normalized by its own sum, plus a log2-sum-exp per head. A combine kernel merges the splits.
- Registers: the forward kernel uses 195 VGPR with no spills (1 workgroup per CU); the combine kernel uses 175 VGPR.
  Check with `hipcc ... --cuda-device-only -S` (see the MoE README for the flags).

## Build and test

aiter JIT-builds the module on first use, with HipKittens headers from `HIP_KITTENS_DIR` (fetched at the pinned
commit by default). `op_tests/test_hk_sparse_mla.py` checks against an fp32 reference: prefill and split-K decode,
masked keys, padded top-k, slots past the pool and a pool past 4 GiB.

Run `python op_tests/test_hk_sparse_mla.py` on gfx950. The suite also checks changed-input graph
replay, fully masked rows, int64 indices outside the int32 range, empty query batches, and invalid inputs.

All tensors must be on the same gfx950 device. KV uses OCP `float8_e4m3fn`; FNUZ is not supported.
The Python wrapper accepts floating-point Q and casts it to OCP FP8. KV can be `[N, D]` or have
singleton dimensions between N and D. The softmax scale must be finite and positive. Output must
be contiguous, 16-byte aligned, and use separate storage from the inputs. Warm the JIT before capture.

## Compare with current AITER

```bash
python op_tests/op_benchmarks/bench_hk_sparse_mla.py --output /tmp/hk-sparse-benchmark.json
```

The benchmark tests FP8 and BF16 compute and the default split count plus 1, 2, 4, 8, 16 and 32.
It checks accuracy before selecting candidates, then confirms the fastest candidates and both defaults
with two fresh input seeds. Cases cover 1 to 16384 queries and 8 MiB / 5 GiB KV pools. All selected
slots are valid, so AITER can use its asynchronous LDS path. HK pads its index rows inside the call.
A 512 MiB buffer evicts cache data before each timed one-call HIP graph replay; eviction is not timed.
Raw samples, rejected candidates and correctness results are saved in JSON.

The benchmark additionally tests the OPUS sparse-prefill kernels with losslessly repacked native
inputs: BF16 Q/KV, or 448 FP8 dimensions with unit E8M0 scales plus 64 BF16 dimensions. Both
represent exactly the same Q/K/V values as the flat FP8 input. The native caches occupy 2x and
1.25x the original bytes, respectively. Query/cache repacking is excluded from their timings,
and the attention sink is disabled. The JSON separately reports the strongest matching-layout
Gluon baseline and the strongest passing candidate across all tested layouts.
