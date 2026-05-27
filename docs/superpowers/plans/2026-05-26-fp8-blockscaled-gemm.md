# FP8 Blockscaled Preshuffle GEMM — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the `fp8_einsum.py` 4D einsum kernel to a 2D `(M,N) = A(M,K) @ B(N,K).T` blockscaled GEMM with DeepGEMM SM100 packed-UE8M0 scales (1×128 on A, 128×128 on B) and a preshuffled B layout `(N//16, K//64, 4, 16, 16)`, targeting gfx950.

**Architecture:** Strip the head-dim and the qz epilogue from `fp8_einsum.py`. Rename `H→gone`, `D→N`, `R→K`, `B→M`. Replace the row-major B loader with a preshuffled-tile loader. Replace the bf16-store epilogue with the same bf16 store but indexed by `(M,N)`. Keep the ping-pong K-loop, the `mfma_scale_f32_16x16x128_f8f6f4` with opsel byte index, the buffer-resource OOB protection, and the autotune wrapper (with an empty `_AUTOTUNE_WINNERS`).

**Tech Stack:** flydsl (MLIR-emitting Python DSL), ROCm 6.x + hipcc, MI355X (gfx950, CDNA4, 160 KB LDS/CU). Venv: `source /opt/venv/bin/activate`.

**Reference source:** `aiter/ops/flydsl/kernels/fp8_einsum.py` (lines cited throughout). **Reference spec:** `docs/superpowers/specs/2026-05-26-fp8-blockscaled-gemm-design.md`.

---

## File Structure

| File | Action | Notes |
|---|---|---|
| `aiter/ops/flydsl/kernels/preshuffle_gemm_blockscaled.py` | **Create** (~700 LOC) | Port of `fp8_einsum.py` with the renames + B preshuffle loader. |
| `aiter/ops/flydsl/kernels/preshuffle_gemm_blockscaled_perf/__init__.py` | **Create** (empty) | Package marker. |
| `aiter/ops/flydsl/kernels/preshuffle_gemm_blockscaled_perf/bench.py` | **Create** (~200 LOC) | `build_inputs(M,N,K)`, per-shape ms/TFLOPS table. |
| `aiter/ops/flydsl/kernels/preshuffle_gemm_blockscaled_perf/autotune.py` | **Create** (~200 LOC) | Grid search over `(tile_m, tile_n, k_unroll)`, writes winners JSON. |
| `op_tests/test_preshuffle_gemm_blockscaled.py` | **Create** (~250 LOC) | `@benchmark()` per shape, fp32 dequant reference, argparse `-m -n -k`. |

---

## Task 1: Scaffold the kernel file with imports + autotune stub

**Files:**
- Create: `aiter/ops/flydsl/kernels/preshuffle_gemm_blockscaled.py`

- [ ] **Step 1: Copy `fp8_einsum.py` lines 1-60 (imports + module docstring header)** to the new file. Delete the qz autotune table block. Replace the docstring with: `"""FP8 blockscaled GEMM: C(M,N) = A(M,K) @ B(N,K).T with DeepGEMM packed-UE8M0 scales. Preshuffled B layout (N//16, K//64, 4, 16, 16) for gfx950."""`.

- [ ] **Step 2: Add empty `_AUTOTUNE_WINNERS: dict[tuple[int, int, int], dict] = {}`** (key is `(M, N, K)`). Per spec: no winners shipped initially.

- [ ] **Step 3: Add `_autotune_lookup(M, N, K) -> dict | None`**: literal copy of `fp8_einsum.py:120-160`'s `_autotune_lookup` but keyed on `(M, N, K)` instead of `(B, H, D, R)`. Returns `None` on miss → caller uses defaults.

- [ ] **Step 4: Commit.**
```bash
git add aiter/ops/flydsl/kernels/preshuffle_gemm_blockscaled.py
git commit -m "feat(blockscaled): scaffold preshuffle GEMM kernel module"
```

---

## Task 2: Port the factory signature with the renames

**Files:**
- Modify: `aiter/ops/flydsl/kernels/preshuffle_gemm_blockscaled.py`
- Reference: `fp8_einsum.py:380-460` (`compile_fp8_einsum_clean_ue8m0` factory signature + validation).

- [ ] **Step 1: Copy `fp8_einsum.py:380-460` into a new function**:
```python
def compile_preshuffle_gemm_blockscaled(
    M: int, N: int, K: int,
    tile_m: int = 128, tile_n: int = 128, k_unroll: int = 2,
    num_warps: int = 4, num_stages: int = 2,
    xcd_supergroup_m: int = 8,
    waves_per_eu: int = 1,
    use_async_dma: bool = True,
):
```
Apply renames inside the body: `B→M`, `H→(delete)`, `D→N`, `R→K`. Delete every `quant_output` / `quant_transpose_scale` parameter and branch. Delete every reference to `arg_z` being bf16-vs-fp8 (it's always bf16 here).

- [ ] **Step 2: Update validation block (was `fp8_einsum.py:410-450`)**:
  - `assert M % tile_m == 0`
  - `assert N % tile_n == 0`
  - `assert K % 512 == 0` (one packed-UE8M0 i32 covers 4 × 128 K-cols per spec section "Scale ABI")
  - `assert tile_n in (128, 256)`
  - `assert tile_m in (64, 128, 256)`
  - Delete the H/D-specific asserts.

- [ ] **Step 3: Verify it imports.**
```bash
source /opt/venv/bin/activate
python -c "from aiter.ops.flydsl.kernels.preshuffle_gemm_blockscaled import compile_preshuffle_gemm_blockscaled; print('ok')"
```
Expected: `ok`. If `NameError` on a stale `H`/`D`/`R`/`B` reference, fix and re-run.

- [ ] **Step 4: Commit.**
```bash
git add -u && git commit -m "feat(blockscaled): port factory signature with M/N/K renames"
```

---

## Task 3: Port LDS allocator + buffer resources

**Files:**
- Modify: `aiter/ops/flydsl/kernels/preshuffle_gemm_blockscaled.py`
- Reference: `fp8_einsum.py:460-560` (LDS allocator with `pong`, `ping`, `sx_slab`, `sy_slab`) and `:600-680` (buffer resources).

- [ ] **Step 1: Copy the LDS allocator block from `fp8_einsum.py:460-560`** but DELETE the `qz_amax_slab` allocation entirely (no qz epilogue here). Keep: `a_ping`, `a_pong`, `b_pong` (B-in-regs needs scratch slot? — review: in `fp8_einsum.py` B is in regs, only A goes to LDS; if so delete `b_pong`), `sx_slab`, `sy_slab`. Rename `sx_slab→sa_slab`, `sy_slab→sb_slab`.

- [ ] **Step 2: Compute LDS byte budget** and assert it stays under 160 KB (per spec section "Geometry & LDS budget"):
```python
total_lds = a_ping_bytes + a_pong_bytes + sa_slab_bytes + sb_slab_bytes
assert total_lds <= 160 * 1024, f"LDS budget {total_lds}B exceeds 160KB"
```

- [ ] **Step 3: Port buffer-resource setup from `fp8_einsum.py:600-680`**. Four `num_records_bytes`:
  - `_c_nrec = M * N * 2`  (bf16)
  - `_a_nrec = M * K * 1`  (fp8)
  - `_b_nrec = N * K * 1`  (fp8, total elements unchanged by preshuffle)
  - `_sa_nrec = M * (K // 512) * 4`  (i32 packed UE8M0)
  - `_sb_nrec = (N // 128) * (K // 512) * 4`
  Delete the old `_z`/`_x`/`_y`/`_sx`/`_sy` variants.

- [ ] **Step 4: Commit.**
```bash
git add -u && git commit -m "feat(blockscaled): port LDS allocator and buffer resources"
```

---

## Task 4: Write failing test for the preshuffled-B loader

**Files:**
- Create: `op_tests/test_preshuffle_gemm_blockscaled.py`

- [ ] **Step 1: Write the test file skeleton + the loader-shape test FIRST** (TDD; full kernel test comes in Task 9):
```python
"""Tests for preshuffle_gemm_blockscaled. Style mirrors op_tests/test_mhc.py."""
import argparse
import pandas as pd
import torch
import aiter.test_common as testc
from aiter.ops.flydsl.kernels.preshuffle_gemm_blockscaled import (
    compile_preshuffle_gemm_blockscaled,
)

def _pack_ue8m0_bytes_to_i32(bytes_t: torch.Tensor) -> torch.Tensor:
    """4 fp8-e8m0 bytes (one per 128-col group) → 1 i32, little-endian."""
    assert bytes_t.dtype == torch.uint8 and bytes_t.shape[-1] == 4
    return (bytes_t[..., 0].to(torch.int32)
        | (bytes_t[..., 1].to(torch.int32) << 8)
        | (bytes_t[..., 2].to(torch.int32) << 16)
        | (bytes_t[..., 3].to(torch.int32) << 24))

def _fp32_to_ue8m0_byte(x_fp32: torch.Tensor) -> torch.Tensor:
    """Take abs amax → round up to next power of 2 → encode exponent as uint8."""
    bits = x_fp32.abs().view(torch.int32)
    exp = ((bits + 0x400000) & 0xFF800000) >> 23
    return exp.clamp(0, 255).to(torch.uint8)

def build_inputs(M: int, N: int, K: int, device: str = "cuda", seed: int = 0):
    """Returns (a_fp8, b_fp8_preshuf, sa_i32, sb_i32, c_ref_bf16)."""
    g = torch.Generator(device=device).manual_seed(seed)
    a_f32 = torch.randn(M, K, device=device, generator=g) * 0.3
    b_f32 = torch.randn(N, K, device=device, generator=g) * 0.3

    # A: per-row 1x128 ue8m0 scale
    sa_amax = a_f32.abs().reshape(M, K // 128, 128).amax(-1)  # (M, K//128)
    sa_byte = _fp32_to_ue8m0_byte(sa_amax)
    sa_pow2 = (2.0 ** (sa_byte.to(torch.int32) - 127)).float()
    a_scaled = a_f32 / sa_pow2.repeat_interleave(128, dim=-1).clamp_min(1e-30)
    a_fp8 = a_scaled.clamp(-448, 448).to(torch.float8_e4m3fn)
    sa_i32 = _pack_ue8m0_bytes_to_i32(sa_byte.reshape(M, K // 512, 4))  # (M, K//512)

    # B: 128x128 ue8m0 scale (one scalar per 128×128 tile)
    b_blk = b_f32.reshape(N // 128, 128, K // 128, 128).permute(0, 2, 1, 3)
    sb_amax = b_blk.abs().amax(dim=(-1, -2))  # (N//128, K//128)
    sb_byte = _fp32_to_ue8m0_byte(sb_amax)
    sb_pow2 = (2.0 ** (sb_byte.to(torch.int32) - 127)).float()
    b_scaled = b_f32 / (sb_pow2
        .repeat_interleave(128, dim=0)
        .repeat_interleave(128, dim=1)
        .clamp_min(1e-30))
    b_fp8_rowmajor = b_scaled.clamp(-448, 448).to(torch.float8_e4m3fn)
    sb_i32 = _pack_ue8m0_bytes_to_i32(sb_byte.reshape(N // 128, K // 512, 4))

    # Preshuffle B: (N, K) -> (N//16, K//64, 4, 16, 16)
    # 4 = 4 packs of 16 K-cols each; inner (16, 16) = (n_in_16, k_in_16)
    b_view = b_fp8_rowmajor.reshape(N // 16, 16, K // 64, 4, 16)  # (N//16, 16, K//64, 4, 16)
    b_preshuf = b_view.permute(0, 2, 3, 1, 4).contiguous()  # (N//16, K//64, 4, 16, 16)

    # fp32 reference
    a_dq = a_fp8.float() * sa_pow2.repeat_interleave(128, dim=-1)
    b_dq = b_fp8_rowmajor.float() * (sb_pow2
        .repeat_interleave(128, dim=0)
        .repeat_interleave(128, dim=1))
    c_ref = (a_dq @ b_dq.T).bfloat16()
    return a_fp8, b_preshuf, sa_i32, sb_i32, c_ref


def test_build_inputs_shapes():
    M, N, K = 256, 256, 512
    a, b, sa, sb, c = build_inputs(M, N, K)
    assert a.shape == (M, K) and a.dtype == torch.float8_e4m3fn
    assert b.shape == (N // 16, K // 64, 4, 16, 16) and b.dtype == torch.float8_e4m3fn
    assert sa.shape == (M, K // 512) and sa.dtype == torch.int32
    assert sb.shape == (N // 128, K // 512) and sb.dtype == torch.int32
    assert c.shape == (M, N) and c.dtype == torch.bfloat16
```

- [ ] **Step 2: Run the test (it tests `build_inputs` only; the import of the kernel module will succeed thanks to Task 2).**
```bash
source /opt/venv/bin/activate
pytest op_tests/test_preshuffle_gemm_blockscaled.py::test_build_inputs_shapes -xvs
```
Expected: PASS.

- [ ] **Step 3: Commit.**
```bash
git add op_tests/test_preshuffle_gemm_blockscaled.py
git commit -m "test(blockscaled): add build_inputs harness with shape assertions"
```

---

## Task 5: Port A-loader + scale loaders (rename only)

**Files:**
- Modify: `aiter/ops/flydsl/kernels/preshuffle_gemm_blockscaled.py`
- Reference helpers in `fp8_einsum.py`:
  - `load_a_tile_to_lds_async`: lines 750-820
  - `load_sx_to_lds` / `prefetch_sx_tile`: lines 880-960
  - `load_sy_to_lds` / `prefetch_sy_tile`: lines 1040-1120

- [ ] **Step 1: Copy `load_a_tile_to_lds_async` (`fp8_einsum.py:750-820`) verbatim.** The A row-major layout is unchanged from the einsum source; only the outer-dim variable renames (`B→M`, `R→K`) apply. Rename the function to `load_a_tile_to_lds_async` (same name OK).

- [ ] **Step 2: Copy `load_sx_to_lds` and `prefetch_sx_tile` → rename to `load_sa_to_lds` and `prefetch_sa_tile`.** Replace every `arg_sx` with `arg_sa`, `_sx_nrec` with `_sa_nrec`, `sx_slab` with `sa_slab`. Stride math: SA is `(M, K//512)` i32, identical layout to the old SX `(B, H, R//512)` after collapsing H — just drop the H stride.

- [ ] **Step 3: Copy `load_sy_to_lds` and `prefetch_sy_tile` → rename to `load_sb_to_lds` and `prefetch_sb_tile`.** Replace `arg_sy` with `arg_sb`, etc. SB shape is `(N//128, K//512)` — collapse the old `(D//128, H, R//512)` shape by deleting the H dim.

- [ ] **Step 4: Commit.**
```bash
git add -u && git commit -m "feat(blockscaled): port A and scale loaders with renames"
```

---

## Task 6: Write the preshuffled-B loader (NEW — no direct analogue)

**Files:**
- Modify: `aiter/ops/flydsl/kernels/preshuffle_gemm_blockscaled.py`
- Reference: `fp8_einsum.py:830-880` (`load_b_tile` — the row-major B loader being **replaced**). Useful for the surrounding `raw_ptr_buffer_load` / `lds_load_b_packs_for_g` plumbing only.

- [ ] **Step 1: Implement `load_b_tile_preshuffled(bx_n, k_tile_idx, b_rsrc, lane_id)`** that loads one `tile_n × 64`-K slab of B (the K-step matches the MFMA K=128 / 2 K-packs per tile already in the einsum kernel) from the preshuffled layout `(N//16, K//64, 4, 16, 16)` directly into registers. Layout math (from spec section "B preshuffle"):
  - Outer index = `bx_n * (tile_n // 16) + n_tile_16` for `n_tile_16 in range(tile_n // 16)`
  - K-outer index = `k_tile_idx` (advances by 1 per K-step of 64)
  - Per-lane: each lane owns 1 of 16 `(n_in_16, k_in_16)` slots per `pack` ∈ [0,4). One `i32` `buffer_load` (4 fp8 bytes) per pack per lane.
  - Total per lane per K-step: `(tile_n // 16) * 4` `i32` loads = e.g. tile_n=128 → 8×4=32 i32 loads.

- [ ] **Step 2: Wire it into the kernel body** in place of the old `load_b_tile` call. Reference call site: search `fp8_einsum.py` for `load_b_tile(` (around line 1320 in the K-loop driver).

- [ ] **Step 3: Sanity-compile.**
```bash
source /opt/venv/bin/activate
python -c "
from aiter.ops.flydsl.kernels.preshuffle_gemm_blockscaled import compile_preshuffle_gemm_blockscaled
launch = compile_preshuffle_gemm_blockscaled(256, 256, 512)
print('compiled')
"
```
Expected: `compiled` (or a clear flydsl MLIR error pointing at the new B-loader code path).

- [ ] **Step 4: Commit.**
```bash
git add -u && git commit -m "feat(blockscaled): preshuffled-B loader for (N//16, K//64, 4, 16, 16) layout"
```

---

## Task 7: Port MFMA + K-loop driver (rename only)

**Files:**
- Modify: `aiter/ops/flydsl/kernels/preshuffle_gemm_blockscaled.py`
- Reference: `fp8_einsum.py:1150-1500` (MFMA `mfma_scale_f32_16x16x128_f8f6f4`, `pack_i64x4_to_i32x8`, `lds_a0_prefetch`, `compute_tile`) and `:1500-1750` (K-loop prologue + unrolled pingpong + odd/even epilogue).

- [ ] **Step 1: Copy `pack_i64x4_to_i32x8`, `lds_load_a_packs_k64`, `lds_load_b_packs_for_g`, `lds_a0_prefetch`, `compute_tile` verbatim** (`fp8_einsum.py:1150-1500`). Inside `compute_tile`, the MFMA call site and the opsel byte-index logic for SA/SB are unchanged — the renames `sx→sa`, `sy→sb` are local to the variable names already done in Task 5.

- [ ] **Step 2: Copy the K-loop driver from `fp8_einsum.py:1500-1750`.** This includes:
  - Prologue: prefetch K=0 A tile to LDS, B-tile to regs, SA/SB scales to LDS.
  - Unrolled pingpong body: 2 inner K-steps per outer iter, with `lds_a0_prefetch` overlapping the next A-pack load with the current MFMA.
  - Odd/even epilogue: drain the pipeline.
  Rename every `quant_output` branch to the bf16 path (delete the qz branch entirely).

- [ ] **Step 3: Copy `store_output_bf16` from `fp8_einsum.py:1380-1450`** as `store_output_bf16(bx_m, bx_n, mi, ii, ni, acc, c_rsrc)`. Replace any `arg_z`/D-stride references with `arg_c`/N-stride. Delete the qz epilogue function entirely.

- [ ] **Step 4: Commit.**
```bash
git add -u && git commit -m "feat(blockscaled): port MFMA, K-loop driver, and bf16 store epilogue"
```

---

## Task 8: Wire `@flyc.kernel` + `@flyc.jit launch` + convenience wrapper

**Files:**
- Modify: `aiter/ops/flydsl/kernels/preshuffle_gemm_blockscaled.py`
- Reference: `fp8_einsum.py:1760-1888` (`@flyc.kernel kernel_gemm`, `@flyc.jit launch_gemm`, `compile_fp8_einsum_clean_ue8m0` factory's tail).

- [ ] **Step 1: Copy the `@flyc.kernel kernel_gemm(...)` decorator block.** New ABI (from spec section "Launcher ABI"):
```python
@flyc.kernel
def kernel_gemm(
    arg_c: fx.Tensor,   # bf16 (M, N)
    arg_a: fx.Tensor,   # fp8  (M, K)
    arg_b: fx.Tensor,   # fp8  (N//16, K//64, 4, 16, 16) preshuffled
    arg_sa: fx.Tensor,  # i32  (M, K//512)
    arg_sb: fx.Tensor,  # i32  (N//128, K//512)
    i32_m: fx.Int32,
):
    ...
```
Delete the old `arg_z, arg_x, arg_y, arg_sx, arg_sy, i32_b` ABI.

- [ ] **Step 2: Copy the `@flyc.jit launch_gemm` wrapper.** New signature:
```python
def launch(arg_c, arg_a, arg_b, arg_sa, arg_sb, i32_m, stream):
    grid = (M // tile_m * N // tile_n,)
    block = (num_warps * 64,)
    kernel_gemm[grid, block, stream](arg_c, arg_a, arg_b, arg_sa, arg_sb, i32_m)
```

- [ ] **Step 3: Add `compile_preshuffle_gemm_blockscaled_auto(M, N, K)` convenience wrapper** (mirrors `fp8_einsum.py:60-120`'s `compile_fp8_einsum_clean_ue8m0_auto`): looks up `_AUTOTUNE_WINNERS` via `_autotune_lookup(M, N, K)`, falls back to defaults, calls `compile_preshuffle_gemm_blockscaled(...)`.

- [ ] **Step 4: Sanity-compile end-to-end.**
```bash
source /opt/venv/bin/activate
python -c "
import torch
from aiter.ops.flydsl.kernels.preshuffle_gemm_blockscaled import compile_preshuffle_gemm_blockscaled_auto
launch = compile_preshuffle_gemm_blockscaled_auto(256, 256, 512)
print('compiled, launch =', launch)
"
```
Expected: prints a callable. Any compile error → fix and re-run.

- [ ] **Step 5: Commit.**
```bash
git add -u && git commit -m "feat(blockscaled): @flyc.kernel + launch + autotune-aware wrapper"
```

---

## Task 9: End-to-end numerical correctness test

**Files:**
- Modify: `op_tests/test_preshuffle_gemm_blockscaled.py`

- [ ] **Step 1: Add the correctness test** at the bottom of the test file:
```python
@testc.benchmark()
def test_blockscaled_gemm_correctness(M=256, N=256, K=512):
    a, b, sa, sb, c_ref = build_inputs(M, N, K)
    c_out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    launch = compile_preshuffle_gemm_blockscaled_auto(M, N, K)
    stream = torch.cuda.current_stream().cuda_stream
    launch(c_out, a, b, sa, sb, M, stream)
    torch.cuda.synchronize()

    err = (c_out.float() - c_ref.float()).abs()
    rel = err / (c_ref.float().abs() + 1e-6)
    return {
        "M": M, "N": N, "K": K,
        "max_abs": err.max().item(),
        "max_rel": rel.max().item(),
        "median_rel": rel.median().item(),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=int, nargs="+", default=[256, 1024, 4096])
    parser.add_argument("-n", type=int, nargs="+", default=[256, 1024, 4096])
    parser.add_argument("-k", type=int, nargs="+", default=[512, 2048, 8192])
    args = parser.parse_args()
    rows = []
    for M in args.m:
        for N in args.n:
            for K in args.k:
                rows.append(test_blockscaled_gemm_correctness(M=M, N=N, K=K))
    print(pd.DataFrame(rows).to_markdown(index=False))
```

- [ ] **Step 2: Add import:** at top of the file, ensure `from aiter.ops.flydsl.kernels.preshuffle_gemm_blockscaled import compile_preshuffle_gemm_blockscaled_auto` is present.

- [ ] **Step 3: Run the test.**
```bash
source /opt/venv/bin/activate
pytest op_tests/test_preshuffle_gemm_blockscaled.py::test_blockscaled_gemm_correctness -xvs
```
Expected: PASS bar = `median_rel < 5e-2`, `max_rel < 0.5` (per spec section "Correctness bar": fp8 + 128-block scaling → ~1-3% typical median).

- [ ] **Step 4: If median_rel > 5%**, the suspects in order: (a) preshuffle perm direction in `build_inputs`, (b) MFMA opsel byte index for SA/SB (verify against `fp8_einsum.py` MFMA call), (c) B-loader pack ordering. Fix and re-run.

- [ ] **Step 5: Commit.**
```bash
git add -u && git commit -m "test(blockscaled): end-to-end correctness vs fp32 dequant reference"
```

---

## Task 10: Bench harness

**Files:**
- Create: `aiter/ops/flydsl/kernels/preshuffle_gemm_blockscaled_perf/__init__.py`
- Create: `aiter/ops/flydsl/kernels/preshuffle_gemm_blockscaled_perf/bench.py`

- [ ] **Step 1: Create empty `__init__.py`.**

- [ ] **Step 2: Write `bench.py`** modeled on `aiter/ops/flydsl/kernels/fp8_einsum_perf/bench_ni_rotation.py`:
```python
"""Bench preshuffle_gemm_blockscaled across (M, N, K) shapes."""
import argparse
import time
import pandas as pd
import torch
from aiter.ops.flydsl.kernels.preshuffle_gemm_blockscaled import (
    compile_preshuffle_gemm_blockscaled_auto,
)
from op_tests.test_preshuffle_gemm_blockscaled import build_inputs

def bench_one(M, N, K, n_iters=50, n_warmup=10):
    a, b, sa, sb, _ = build_inputs(M, N, K)
    c_out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    launch = compile_preshuffle_gemm_blockscaled_auto(M, N, K)
    stream = torch.cuda.current_stream().cuda_stream
    for _ in range(n_warmup):
        launch(c_out, a, b, sa, sb, M, stream)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        launch(c_out, a, b, sa, sb, M, stream)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n_iters * 1000
    tflops = 2 * M * N * K / (ms * 1e-3) / 1e12
    return {"M": M, "N": N, "K": K, "ms": round(ms, 4), "TFLOPS": round(tflops, 1)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("-n", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("-k", type=int, nargs="+", default=[2048, 8192])
    args = parser.parse_args()
    rows = [bench_one(M, N, K) for M in args.m for N in args.n for K in args.k]
    print(pd.DataFrame(rows).to_markdown(index=False))
```

- [ ] **Step 3: Smoke-run.**
```bash
source /opt/venv/bin/activate
python -m aiter.ops.flydsl.kernels.preshuffle_gemm_blockscaled_perf.bench -m 1024 -n 1024 -k 2048
```
Expected: one-line markdown table with ms + TFLOPS.

- [ ] **Step 4: Commit.**
```bash
git add aiter/ops/flydsl/kernels/preshuffle_gemm_blockscaled_perf/
git commit -m "perf(blockscaled): bench harness with ms + TFLOPS table"
```

---

## Task 11: Autotune harness

**Files:**
- Create: `aiter/ops/flydsl/kernels/preshuffle_gemm_blockscaled_perf/autotune.py`

- [ ] **Step 1: Write `autotune.py`** modeled on the autotune patterns in `aiter/ops/flydsl/kernels/fp8_einsum_perf/` (search for `autotune` files there):
```python
"""Grid search over (tile_m, tile_n, k_unroll) for preshuffle_gemm_blockscaled."""
import argparse
import itertools
import json
import time
import pandas as pd
import torch
from aiter.ops.flydsl.kernels.preshuffle_gemm_blockscaled import (
    compile_preshuffle_gemm_blockscaled,
)
from op_tests.test_preshuffle_gemm_blockscaled import build_inputs

GRID = list(itertools.product(
    [64, 128, 256],   # tile_m
    [128, 256],       # tile_n
    [1, 2, 4],        # k_unroll
))

def autotune_one(M, N, K, n_iters=30, n_warmup=5):
    a, b, sa, sb, _ = build_inputs(M, N, K)
    c_out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    stream = torch.cuda.current_stream().cuda_stream
    best = None
    for tm, tn, ku in GRID:
        if M % tm or N % tn: continue
        try:
            launch = compile_preshuffle_gemm_blockscaled(M, N, K, tile_m=tm, tile_n=tn, k_unroll=ku)
        except AssertionError:
            continue  # LDS budget or other validation failed
        for _ in range(n_warmup):
            launch(c_out, a, b, sa, sb, M, stream)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            launch(c_out, a, b, sa, sb, M, stream)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / n_iters * 1000
        if best is None or ms < best["ms"]:
            best = {"M": M, "N": N, "K": K, "tile_m": tm, "tile_n": tn, "k_unroll": ku, "ms": round(ms, 4)}
    return best

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("-n", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("-k", type=int, nargs="+", default=[2048, 8192])
    parser.add_argument("--out", default="autotune_winners.json")
    args = parser.parse_args()
    winners = [autotune_one(M, N, K) for M in args.m for N in args.n for K in args.k]
    print(pd.DataFrame(winners).to_markdown(index=False))
    with open(args.out, "w") as f:
        json.dump(winners, f, indent=2)
    print(f"wrote {args.out}")
```

- [ ] **Step 2: Smoke-run at a single tiny shape.**
```bash
source /opt/venv/bin/activate
python -m aiter.ops.flydsl.kernels.preshuffle_gemm_blockscaled_perf.autotune -m 1024 -n 1024 -k 2048
```
Expected: prints a winner row + writes `autotune_winners.json`.

- [ ] **Step 3: Commit.**
```bash
git add aiter/ops/flydsl/kernels/preshuffle_gemm_blockscaled_perf/autotune.py
git commit -m "perf(blockscaled): autotune harness with JSON winners output"
```

---

## Task 12: Final smoke pass + plan-completion checklist

- [ ] **Step 1: Run all tests.**
```bash
source /opt/venv/bin/activate
pytest op_tests/test_preshuffle_gemm_blockscaled.py -xvs
```
Expected: all green.

- [ ] **Step 2: Bench a couple production-relevant shapes.**
```bash
python -m aiter.ops.flydsl.kernels.preshuffle_gemm_blockscaled_perf.bench -m 4096 -n 4096 -k 8192
```
Record ms + TFLOPS in the commit message of the final commit.

- [ ] **Step 3: Final commit (or push branch + open PR per project convention).**

---

## Self-Review Notes (not for the implementer — for the planner)

Spec sections covered:
- Launcher ABI → Task 8
- Tensor shapes (C, A, B, SA, SB) → Task 4 (build_inputs) + Task 8 (kernel sig)
- Scale ABI (packed UE8M0 i32) → Task 4 (`_pack_ue8m0_bytes_to_i32`) + Task 5 (loaders)
- B preshuffle `(N//16, K//64, 4, 16, 16)` → Task 4 (host-side perm) + Task 6 (device-side loader)
- 15-step implementation order → tasks 1-12 collapse near-renames into single tasks; the spec's items 1-4 are Task 1-2, items 5-8 are Tasks 3-5, items 9-11 are Tasks 6-7, items 12-13 are Task 8, items 14-15 are Tasks 9-12
- LDS budget assertion → Task 3 step 2
- fp32 dequant reference → Task 4 step 1 (last block of `build_inputs`)
- Empty `_AUTOTUNE_WINNERS` → Task 1 step 2
- Correctness bar (median_rel < 5%) → Task 9 step 3
- test_mhc.py-style scaffolding (`@benchmark()`, argparse `-m -n -k`, `pd.DataFrame.to_markdown`) → Tasks 4, 9, 10, 11
- gfx950 / 160 KB LDS → Task 3 step 2 assertion

No placeholders. Type/name consistency checked: `arg_c/arg_a/arg_b/arg_sa/arg_sb/i32_m` used identically in Task 8, Task 9, Task 10, Task 11. `compile_preshuffle_gemm_blockscaled` vs `compile_preshuffle_gemm_blockscaled_auto` distinction held throughout (auto = lookup wrapper; non-auto = explicit factory). `build_inputs(M, N, K)` signature identical in Tasks 4, 9, 10, 11.
