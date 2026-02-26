# 🔧 Assembly Programming & Loading Buffers to LDS on CDNA

[← Kernel Optimization](kernel_optimization.md)

---

This section explains how to implement efficient memory buffer loads into Local Data Share (LDS) using assembly on AMD’s **CDNA** architecture.

> **Scope:** CDNA architecture. This guide uses conceptual, practical, and pseudo-assembly examples that are directly applicable to CDNA‑class devices. For exact opcode names and encodings, consult the CDNA ISA manual and assembler references for your specific GPU generation.

---

## 🧭 Overview & Motivation

Staging data from global memory into LDS (shared memory) is a central optimization for many high-performance kernels (GEMM, convolution, stencils). Doing this efficiently in assembly gives you precise control over:

* Instruction scheduling and latency hiding
* Vectorized/coalesced memory loads
* Register usage and packing into VGPRs
* LDS layout and bank conflict avoidance

Assembly-level control is useful for microkernels where every cycle counts; it’s most valuable when you need deterministic scheduling, exact register allocation, or to use instructions not exposed by higher-level languages.

## 🧱 Typical workflow (high level)

1. Compute each thread’s global source address for its portion of the tile.
2. Perform **coalesced VMEM loads** into VGPRs (use vectorized loads when possible).
3. Optionally perform address-to-LDS-index conversion (compute where each element should reside in LDS).
4. Write VGPRs into LDS using `ds_write` instructions (per-element or vector stores).
5. Use `s_barrier` to ensure all LDS stores are visible.
6. Threads read from LDS (`ds_read`) for the compute phase.

This sequence is repeated for each tile along the K dimension in GEMM-like loops.

---

## 🛠 Assembly primitives (conceptual)

Below are the common classes of instructions you’ll use. Mnemonics may differ slightly across assemblers; the names below follow common AMD assembly conventions:

* **VMEM (buffer) loads:** `buffer_load_dword`, `buffer_load_dwordx4`, `buffer_load_b128`, etc. — load 32-bit words, vectors, or 128‑bit blocks from global memory into VGPRs.
* **VGPR arithmetic:** `v_add_u32`, `v_lshlrev_b32`, `v_mul_lo_u32`, etc. — compute addresses and indices in VGPRs.
* **LDS stores/loads:** `ds_write_b32`, `ds_write_b64`, `ds_read_b32`, `ds_read_b64` — write/read to/from LDS at byte offsets computed in VGPRs.
* **Scalar waits/barriers:** `s_waitcnt vmcnt(0)` to wait for VMEM to complete, `s_barrier` for intra-workgroup synchronization.
* **Synchronization for LDS:** `s_barrier` (ensures all work-group LDS writes are visible before reads).

**Note:** Use `s_waitcnt` before `s_barrier` where you need to ensure memory loads have completed; `buffer_load` may be issued as `offen` (offset enable) with `s_waitcnt vmcnt(0)` to wait for outstanding VMEM operations.

---

## 🔁 Example pattern — Cooperative tile load (pseudo-assembly)

This pattern assumes a `TILE_W × TILE_H` tile of elements, with each thread responsible for one or more elements. It demonstrates how threads cooperatively load a tile into LDS.

**Assumptions:**

* `waveSize` = lanes per wave (e.g., 64)
* `elem_size` = 4 (bytes) for `float`
* Global base pointers are in scalar registers (`s_addrA`, `s_strideA`)
* Per-thread VGPRs: `v0, v1, ...` used as temporaries

```cpp
// Pseudo-assembly: cooperative load into LDS
// Each thread computes its linear index in the tile
v_mov_b32 vThreadId, v0         // thread lane id (precomputed)
// Compute global address: base + (tile_row * row_stride + tile_col) * elem_size
// For multiple elements per thread, loop over element idx

// --- Phase 1: Issue VMEM loads (vectorized where possible) ---
buffer_load_dwordx4 vA0, vAddr, s[s_addrA:offset], soffset(off)
// or buffer_load_b128 vA0, vAddr, s[s_addrA:offset], offen

// Optionally wait for VMEM to complete before LDS write
s_waitcnt vmcnt(0)

// --- Phase 2: Compute LDS offsets (per-thread) ---
v_mul_u32_u24 vLdsOff, vThreadId, elem_size
v_lshlrev_b32 vLdsOffBytes, shift_amt, vLdsOff // multiply by element size if needed

// --- Phase 3: Store into LDS ---
ds_write_b32 vLdsOffBytes, vA0
// For 4x loads, use ds_write_b128 / multiple ds_write_b32

// --- Phase 4: Synchronize ---
s_barrier

// Now LDS contains the tile; threads can read via ds_read_* for compute

// Example read
ds_read_b32 vLoadVal, vLdsOffBytes
```

**Notes:**

* Group VMEM loads so that multiple lanes read contiguous memory ranges — use `buffer_load_dwordx4` or 128-bit loads where alignment allows.
* Use `s_waitcnt vmcnt(0)` before reading LDS or before `s_barrier` if loads were `offen` and not guaranteed complete.
* If VMEM instructions use `offen`, they can be issued without immediate wait, enabling memory/computation overlap; but ensure completion with `s_waitcnt` before consuming data.

---

## 🧠 Address calculation and VGPR packing

Efficient address calculations and packing of loaded data into VGPRs will make the LDS writes straightforward and avoid extra shuffles.

### Tips

* Compute base addresses in scalar (`s`) registers when possible to reduce per-lane computation.
* Use vectorized VMEM loads to obtain multiple contiguous elements in a single VGPR (e.g., `vA0` contains 4 floats after `buffer_load_dwordx4`).
* If data must be transposed when written to LDS (e.g., for column-major to row-major conversion), perform lightweight VGPR shuffles before `ds_write`.

---

## 🧮 LDS layout and bank conflict avoidance

LDS is banked. To avoid bank conflicts:

* **Pad row strides**: add 1 or a small padding factor to the row stride in LDS to prevent multiple threads accessing addresses that map to the same bank.
* **Align per-row start**: ensure the starting offset of each row is not a multiple of the bank count multiplied by element size.
* **Interleave carefully**: when mapping 2D tiles to linear LDS addresses, choose a layout that maps adjacent threads to different banks.

**Example:**

```cpp
// C-like LDS declaration with padding
__shared__ float tile[TILE_H][TILE_W + PAD]; // PAD chosen to avoid bank aliasing
```

When writing LDS in assembly, compute `lds_offset = row*(TILE_W + PAD) + col` before performing `ds_write_b32`.

---

## ⏱ Latency hiding & scheduling

To hide VMEM and LDS latencies:

* **Issue VMEM loads early** (prefetch next tile) and do useful arithmetic while loads are outstanding.
* **Interleave loads and stores** where possible: while waiting for VMEM, perform address computations or issue independent ALU ops.
* **Use multiple waves** per CU so that when one wave stalls, another can execute; avoid code patterns that reduce wave concurrency (e.g., excessive registers).

Example schedule:

1. Issue `buffer_load` for next chunk
2. Compute addresses for subsequent lanes
3. `s_waitcnt vmcnt(0)`
4. `ds_write` to LDS
5. `s_barrier`
6. Compute using LDS data

---

## 🚀 MI300 and MI350 asynchronous LDS load/store

Some CDNA parts can load directly from HBM into LDS (and store back) without an intermediate VMEM staging step. This can reduce VGPR pressure and simplify certain microkernels, but it is size-limited and still requires explicit synchronization.

**Supported transfer sizes:**

* **MI300:** 1, 2, 4 bytes
* **MI350:** 1, 2, 4, 8, 16 bytes

**Example (16-byte load on MI350):**

```cpp
__device__ __forceinline__ void
cp_async_zfill_cg(__shared__ void* smem_ptr, void const* global_ptr) {
  __builtin_amdgcn_global_load_lds(src_ptr, global_ptr, 16, 0, 0);
}
```

**Notes:**

* The synchronization counter for async LDS operations is `vmcnt`.
* These intrinsics carry memory side effects; if multiple threads in a block access the same LDS region, use a block-wide sync (e.g., `s_barrier`) before consuming the data.
* There is a compiler bug that blocks template arguments in these intrinsics: [llvm-project#175767](https://github.com/llvm/llvm-project/pull/175767).

```cpp
template<int kByteSize>
__device__ __forceinline__ void
cp_async_zfill_cg(__shared__ void* smem_ptr, void const* global_ptr) {
  // Using kByteSize here is a compile time error
  __builtin_amdgcn_global_load_lds(src_ptr, global_ptr, kByteSize, 0, 0);
}
```

The fix is upstream; until ROCm 7.10, use a literal byte size (e.g., 16) instead of a template parameter.

---

## ⚠️ Common pitfalls

* **Forgetting `s_waitcnt` / `s_barrier`:** reading LDS before VMEM/DS writes complete leads to wrong data or race conditions.
* **Misaligned vector loads:** attempting `b128` loads on unaligned addresses causes extra transactions or low efficiency.
* **Bank conflicts in LDS:** failing to pad rows leads to serialized accesses.
* **Excessive register usage:** too many VGPRs or accumulators reduce occupancy and hide less latency.


## 🧾 Final notes

Assembly-level buffer staging to LDS on CDNA is powerful but intricate. Start from a working high-level kernel, validate each step in small increments, and use profiling tools to guide optimizations. Carefully manage alignment, vector loads, and LDS layout — these are the factors that most influence performance on CDNA.

---
← Kernel Optimization