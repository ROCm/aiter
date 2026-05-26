# Design — FP8 blockscaled GEMM (preshuffled B, DeepGEMM-style UE8M0 scales)

**Status:** Approved (pending user review of this written spec)
**Target:** gfx950 / MI355X
**File:** `aiter/ops/flydsl/kernels/preshuffle_gemm_blockscaled.py`
**Date:** 2026-05-26

---

## Goal

Build a standalone fp8 blockscaled GEMM kernel that ports the validated pipeline
from `fp8_einsum.py` (B-in-regs double-buffered, A-in-LDS ping-pong async DMA,
packed UE8M0 scales fed directly to MFMA via opsel) to a generic `C = A @ B`
shape. M is dynamic; N and K are compile-time.

Out-of-scope (explicitly): qz fp8 output, bias/activation epilogue, cshuffle
epilogue, B-in-LDS variant, MoE dispatch, fp16 output, shipping an autotune
winners table (table starts empty; sweep script ships in-tree).

---

## ABI

### Compile-time constants
- `N: int` — output column count (multiple of `tile_n`)
- `K: int` — contraction depth (multiple of `tile_k`, of `128`, and of `512`)
- `tile_m, tile_n, tile_k: int` — MFMA tile shape
- `block_swizzle_n: int = 0` — L2 supergroup swizzle factor (0 disables)

### Runtime args (launcher signature)
```python
launch(arg_c, arg_a, arg_b, arg_sa, arg_sb, i32_m, stream)
```

### Tensor shapes & dtypes

| Arg | Shape | Dtype | Notes |
|---|---|---|---|
| `arg_c` | `(M, N)` | bf16 | output |
| `arg_a` | `(M, K)` | fp8 e4m3 | caller pre-quantizes |
| `arg_b` | preshuffled (see below) | fp8 e4m3 | caller pre-shuffles ONCE at model load |
| `arg_sa` | `(M, K // 512)` | int32 | 4 packed UE8M0 bytes per i32 along K |
| `arg_sb` | `(N // 128, K // 512)` | int32 | same packing along K |
| `i32_m` | scalar | i32 | runtime M |

### B preshuffle layout

Physical layout (compile-time-known strides since N is static):
```
B physical shape: (N // 16, K // 64, klane=4, nlane=16, kpack=16)  fp8
```

Same as `fp8_einsum.py`'s Y minus the leading head dim. Host-side preshuffle:
```python
b_pre = shuffle_weight(b_fp8, layout=(16, 32))   # aiter helper
```

### Scale convention (DeepGEMM SM100 packed UE8M0)

- One UE8M0 byte per K-128 block, packed 4-to-an-i32 along K (so K dim
  ranges over `K // 512` in the i32 view).
- `sa[m, k_packed]` covers rows `m`, K-128 blocks `k_packed*4 .. k_packed*4+3`.
- `sb[n128, k_packed]` covers N-128 block `n128`, same K-128 range.
- Scales are fed directly to `mfma_scale_f32_16x16x128_f8f6f4` via the opsel
  byte index (which of the 4 bytes in the packed i32) — no fp32 promote.

### Validation rules (factory-build-time)
- `tile_k % 128 == 0`
- `tile_m >= 16`, `tile_m % 16 == 0`
- `tile_n % 128 == 0`, `tile_n >= 128`
- `K % tile_k == 0`, `N % tile_n == 0`
- `K % 512 == 0` (packed UE8M0)
- `(tile_m * (K // 512)) % 256 == 0` (clean sx slab distribution)
- gfx950 only (`get_hip_arch().startswith("gfx95")`)

---

## Pipeline architecture (verbatim port of fp8_einsum v2 pingpong)

### Wave / lane decomposition
- `total_threads = 256`, `wave_size = 64`, `num_waves = 4`
- MFMA: `mfma_scale_f32_16x16x128_f8f6f4`
- `m_repeat = tile_m // 16`
- `n_per_wave = tile_n // 4`
- `num_acc_n = n_per_wave // 16`
- `k_unroll = tile_k // 32` (K=32 MFMAs per tile)
- `groups_per_tile = tile_k // 128`
- `mfmas_per_group = 4`

### Per-K-tile pipeline

1. **Async A load**: `raw_ptr_buffer_load_lds` (gmem → LDS direct,
   swizzle-the-source). Zero A registers carried across iterations.
2. **Sync B load**: `buffer_load vec4 i32` per lane → fp8x16 → i64 packs.
   Carried across iterations via `b_pong`/`b_ping` register lists. **This is
   the "double B register buffer."**
3. **Sync sx/sy load**: packed i32 read once per K-tile, indexed per
   (mi, g) / (ni, g).
4. **MFMA**: `compute_tile(accs, b_tile, kt, lds_a, sx_per_mi, sy_per_ni,
   a0_prefetch)`. For each g, for each mi, for each ni:
   one `mfma_scale_f32_16x16x128_f8f6f4` call with the per-(mi, g) sx i32
   and per-(ni, g) sy i32 fed as opsel-indexed scales.
5. **Single waitcnt+barrier per half**: `s_waitcnt(vmcnt=num_b_loads,
   lgkmcnt=0)` then `gpu.barrier()`. Drains LDS-A async; keeps B vmem in flight.

### Unrolled pingpong K-loop driver

```
prologue:
    issue async A tile 0 → lds_a_pong
    sync-load B tile 0 → b_pong
    barrier
    prefetch sx_pong, sy_pong, a0_pong

loop kt_base in 0..num_tiles-2 step 2:
    # half: pong → ping
    issue async A tile (kt_base+1) → lds_a_ping
    b_ping = load B tile (kt_base+1)
    prefetch sx_ping, sy_ping
    compute_tile(b_pong, lds_a_pong, sx_pong, sy_pong, a0_pong)
    s_waitcnt(vmcnt=N_b, lgkmcnt=0); barrier
    a0_ping = prefetch from lds_a_ping

    # half: ping → pong
    issue async A tile (kt_base+2) → lds_a_pong
    b_pong = load B tile (kt_base+2)
    prefetch sx_pong, sy_pong
    compute_tile(b_ping, lds_a_ping, sx_ping, sy_ping, a0_ping)
    s_waitcnt(vmcnt=N_b, lgkmcnt=0); barrier
    a0_pong = prefetch from lds_a_pong

epilogue: handles (num_tiles % 2) cases (1 tile, even tail, odd tail)
```

Per-iter properties:
- Zero A regs across iters (A always in LDS)
- B in regs double-buffered
- One barrier per K-tile (not two)
- `a0_prefetch`: first A pack of NEXT tile pre-loaded from LDS → regs before
  next-iter MFMA fires → saves 1 ds_read from inner-loop critical path

### LDS layout (per WG)

| Slab | Size | Purpose |
|---|---|---|
| A fp8 pong | `tile_m * tile_k` bytes | ping-pong stage 0 |
| A fp8 ping | `tile_m * tile_k` bytes | ping-pong stage 1 |
| sx slab | `tile_m * (K/512) * 4` bytes | per-WG sx, loaded once at start |
| sy slab | `(tile_n/128) * (K/512) * 4` bytes | per-WG sy, loaded once at start |
| **Total** | bounded ≤ 160 KB | enforced at factory build |

Same allocator pattern as `fp8_einsum.py`: two `SmemAllocator` instances with
named global symbols `smem0`/`smem1` for ping/pong.

### Block swizzle

`xcd_remap_bx_by(bx_raw, by_raw, ..., xcd_swizzle=block_swizzle_n)` from
`mfma_preshuffle_pipeline.py` — identical to fp8_einsum. Reorders WG (bx, by)
mapping for better L2 reuse. `block_swizzle_n=0` disables.

### Grid

```
gx = ceil(M / tile_m)   # M-tiles
gy = N // tile_n        # N-tiles
grid = (gx, gy, 1)
block = (256, 1, 1)
```

No head loop. `bx_m = bx * tile_m`, `by_n = by * tile_n`.

### Buffer-resource OOB protection
- `_a_nrec = M * K` bytes — OOB rows (last tile when M < tile_m * gx) get 0
- `_c_nrec = M * N * 2` bytes — OOB row stores dropped by HW
- `_sa_nrec = M * (K/512) * 4` bytes — OOB sa loads return 0
- B / sb use `max_size=True` (caller-bounded; both are static-N)

### Epilogue (`store_output_bf16`)

Default `mfma_epilog(use_cshuffle=False, body_row=...)` from `mfma_epilogues.py`.
Per (mi, ii, ni): `acc → bf16 cast → buffer_store(2 bytes)`. 32 stores per
lane. No cshuffle (measured non-beneficial for tn=128 in fp8_einsum).

---

## Autotune scaffolding

### Table format
```python
# Key:   (N, K, M) → (tile_m, tile_n, tile_k, block_swizzle_n)
_AUTOTUNE_WINNERS: dict[tuple[int, int, int], tuple[int, int, int, int]] = {
    # Initially empty. Populate by running:
    #   python aiter/ops/flydsl/kernels/preshuffle_gemm_blockscaled_perf/autotune.py -n N -k K
    # Paste output here.
}
```

### Lookup helper

`_autotune_lookup(table, N, K, M)`:
1. Exact `(N, K, M)` hit → return.
2. If `(N, K)` has any tabulated M, prefer the smallest M ≥ requested
   (next-higher); else largest tabulated M for that `(N, K)`.
3. If `(N, K)` not in table at all → `ValueError` listing tuned `(N, K)` pairs.

### Auto wrapper
```python
def compile_preshuffle_gemm_blockscaled_auto(
    *, N: int, K: int, M: int | None = None,
):
    (tm, tn, tk, bsw), _ = _autotune_lookup(_AUTOTUNE_WINNERS, N, K, M)
    return compile_preshuffle_gemm_blockscaled(
        N=N, K=K, tile_m=tm, tile_n=tn, tile_k=tk, block_swizzle_n=bsw,
    )
```

---

## Files to create

| File | Purpose | Approx LOC |
|---|---|---|
| `aiter/ops/flydsl/kernels/preshuffle_gemm_blockscaled.py` | The kernel + autotune scaffolding | ~700 |
| `op_tests/test_preshuffle_gemm_blockscaled.py` | Unit test (style: `op_tests/test_mhc.py`) | ~250 |
| `aiter/ops/flydsl/kernels/preshuffle_gemm_blockscaled_perf/__init__.py` | Empty marker | 1 |
| `aiter/ops/flydsl/kernels/preshuffle_gemm_blockscaled_perf/bench.py` | Standalone verify + bench harness | ~200 |
| `aiter/ops/flydsl/kernels/preshuffle_gemm_blockscaled_perf/autotune.py` | Per-shape autotune sweep, emits dict literal | ~200 |

### Test file structure (mirrors `op_tests/test_mhc.py`)

- Uses `aiter.test_common.{checkAllclose, benchmark, run_perftest}`
- Argparse: `-m`, `-n`, `-k` lists; iterate Cartesian product
- `@benchmark()` decorator on `test_gemm_blockscaled(M, N, K)`
- Reference: **fp32 dequant + matmul**. Specifically:
  ```python
  a_dq = a_fp8.float() * sa_pow2.repeat_interleave(128, dim=-1)   # (M, K)
  b_dq = b_fp8.float() * sb_pow2.repeat_interleave(128, dim=-1) \
                                .repeat_interleave(128, dim=0)    # (N, K)
  c_ref = (a_dq @ b_dq.T).bfloat16()                              # (M, N) bf16
  ```
  Cast to bf16 at the end so we compare like-for-like with the kernel's bf16
  output. The kernel's MFMA accumulates in fp32 internally; the bf16 cast on
  both sides isolates pure quantization error from accumulation-order noise.
- Pass bar: `checkAllclose` defaults (rtol/atol ≈ 1e-2)
- Single fixed tile by default: `(128, 128, 256) bsw=4`
- Pandas markdown summary at the end (per-shape `{err, us, TFLOPS, %peak}`)

### Bench harness (`preshuffle_gemm_blockscaled_perf/bench.py`)

- Builds inputs (its own `build_inputs(M, N, K, device, seed)` analog of
  `bench_ni_rotation.build_inputs`)
- Runs the kernel, verifies against fp32 reference, prints per-shape ms/TF/% peak
- Used by `autotune.py` as the inner-loop bench primitive

### Autotune sweep (`preshuffle_gemm_blockscaled_perf/autotune.py`)

- 64-config grid: `tile_m ∈ {16, 32, 64, 128} × tile_n ∈ {128, 256} ×
  tile_k ∈ {128, 256} × bsw ∈ {0, 2, 4, 8}`
- Static-validity filter (LDS budget, divisibility) before compile
- Pre-compile all valid configs once; per-M, quick-bench (8 warmup / 60 iters)
  to pick winner; precise-bench (20 / 300) the winner
- Output: a Python dict literal ready to paste into `_AUTOTUNE_WINNERS`
- Argparse: `-n`, `-k`, `-m` list to control sweep scope

---

## Dependencies (all confirmed to exist before implementation)

| Helper | Source | Notes |
|---|---|---|
| `shuffle_weight(b, layout=(16, 32))` | aiter | used by `bench_ni_rotation.build_inputs` |
| `mfma_epilog` | `mfma_epilogues.py` | dispatcher; we use `use_cshuffle=False` path |
| `swizzle_xor16`, `tile_chunk_coord_i32`, `xcd_remap_bx_by` | `mfma_preshuffle_pipeline.py` | bit-for-bit same as fp8_einsum |
| `SmemAllocator`, `SmemPtr` | `flydsl.utils.smem_allocator` | same allocator pattern |
| `buffer_ops`, `gpu`, `range_constexpr`, `rocdl`, `const_expr` | `flydsl.expr` | standard |
| `mfma_scale_f32_16x16x128_f8f6f4` | `flydsl.expr.rocdl` | the MFMA instruction |
| `raw_ptr_buffer_load_lds` | `flydsl.expr.rocdl` | async DMA primitive |
| `_fp32_to_ue8m0_byte`, `_pack_ue8m0_bytes_to_i32`, `_per_128k_amax` | `bench_ni_rotation.py` | input quant helpers for test/bench |
| `aiter.test_common.{checkAllclose, benchmark, run_perftest}` | aiter | used by `test_mhc.py` |

---

## Out of scope (explicit)

- **No qz fp8 output** (bf16-only); add later if a downstream consumer needs it
- **No cshuffle epilogue** (per Section 2; not a measured win for tn=128)
- **No bias / activation** (per Q7)
- **No B-in-LDS variant** (B always in regs per user's ask)
- **No MoE dispatch** (single GEMM only)
- **No fp16 output** (bf16 only)
- **No autotune winners shipped** (table starts empty; sweep script in-tree)

---

## Risk / unknowns

- The pipeline is a verbatim port of validated code (~1900 TF peak on
  fp8_einsum). Main risk is in the stride/index changes from removing
  the head dim — bx_h, y_head_byte_off, stride_b_x = H*R, stride_b_z = H*D
  all need careful elimination. Mitigated by side-by-side review against
  `fp8_einsum.py` during implementation.
- The test's fp32 reference needs the inverse of the `shuffle_weight`
  preshuffle (un-shuffle B back to row-major for the matmul). Confirm
  the inverse exists in aiter, or write it inline in the test.

---

## Implementation order (preview, for the writing-plans skill)

1. Stand up the empty file with imports + factory signature + validation
2. LDS allocator setup (A pong/ping + sx slab + sy slab) + budget check
3. Kernel body (`_kernel_body`) sketch: bx/by/wave/lane decomp
4. A async load + sx load helpers (reuse fp8_einsum's, retarget strides)
5. B sync load + sy load helpers
6. compute_tile (scaled MFMA, no promote)
7. a0_prefetch helper
8. K-loop driver (prologue + unrolled pingpong + epilogue handling)
9. `store_output_bf16` epilogue
10. `@flyc.kernel kernel_gemm` wrapper + `@flyc.jit launch_gemm` host launcher
11. `_AUTOTUNE_WINNERS` empty dict + `_autotune_lookup` helper + `_auto` wrapper
12. Bench harness scaffold (`preshuffle_gemm_blockscaled_perf/bench.py`)
13. Autotune script (`preshuffle_gemm_blockscaled_perf/autotune.py`)
14. Unit test (`op_tests/test_preshuffle_gemm_blockscaled.py`) — fp32-ref correctness
15. Smoke-test: compile + launch at one shape, assert checkAllclose passes
