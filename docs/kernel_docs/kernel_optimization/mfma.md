# 🛠 Assembly Programming & MFMA

[← Kernel Optimization](kernel_optimization.md)

---

This section explains how to approach low‑level assembly programming on AMD GPUs specifically to use matrix‑multiply/accumulate instructions (MFMA) for very high‑throughput linear algebra kernels. It covers the GPU ISA basics, how MFMA works conceptually, programming patterns (tile & register layout), writing / embedding assembly, verification and debugging, and performance tuning.

> Note: AMD’s matrix instructions are architecture-specific (GFX9/RDNA/GCN → MFMA variants). This guide focuses on concepts, safe patterns, and practical steps you can apply across AMD families. For exact opcode names and encodings, consult your target architecture's ISA manual and the LLVM/ROCm sources.

---

## 🎯 Why write assembly and use MFMA?

* **Peak performance:** MFMA (matrix fused multiply-add) instructions perform many multiply‑accumulate operations per instruction and map directly to the hardware matrix pipelines, delivering extremely high FLOP/s when used correctly.
* **Fine-grained control:** Assembly lets you control register allocation, instruction scheduling, and memory/register traffic precisely — useful for microkernels (GEMM micro-kernels) that dominate performance-critical codes.
* **Reduced instruction overhead:** MFMA reduces instruction count and increases arithmetic intensity compared to hand-coded loops of scalar FMAs.

Use assembly + MFMA when you need the last 10–30% of performance in a compute‑bound kernel — otherwise high‑quality compiler intrinsics and tuned libraries (rocBLAS, Tensile) are preferable.

---

## ⚙️ Prerequisites & toolchain

* **Hardware/ISA docs:** Download the ISA / Programmer’s Reference for your GPU family (GFX9/GFX10/RDNA) from AMD or vendor sources.
* **ROCm toolchain:** `hipcc`, LLVM/Clang with AMDGPU backend, `llvm-mc`, `objdump` / `llvm-objdump`, and device-libs.
* **Assembler support:** LLVM’s assembler (llvm-mc) or inline assembly via HIP/Clang. You may also generate binary encoding via `.inst` pseudo-op in inline assembly or use compiler intrinsics if available.
* **Profiler & validator:** Radeon GPU Profiler (RGP), `rocprof`, and unit tests to validate correctness and measure performance.

---

## 🧭 MFMA — Conceptual model

* **What MFMA does:** A single MFMA instruction computes a small matrix multiply‑accumulate operation: it multiplies a tile block of elements from two source registers (or register groups) and accumulates into destination accumulators. Typical shapes might be 4×4, 8×8, 16×16 micro‑tiles (architecture dependent).

* **Accumulator registers:** MFMA writes results into special accumulator registers (often a register file area or special dest registers). You must plan the accumulator layout so results map correctly to the output matrix tile.

* **Operand packing:** Input operands are packed in vector registers in a layout that matches the MFMA operand shape. You typically pre-load and rearrange data into v-registers (vector regs) or LDS so MFMA consumes them efficiently.

* **Latency/throughput:** MFMA is a relatively long‑latency instruction but very high throughput when many wavefronts and instruction-level parallelism (ILP) are available. The instruction also may have specific scheduling constraints (e.g., wait counts).

---

## 🧱 High-level implementation strategy (GEMM microkernel)

1. **Choose micro‑tile sizes** (m_reg × n_reg output per thread) that match MFMA shapes and fit in registers. Typical micro-tiles: small (e.g., 4×4) per thread but aggregated across threads/waves to form the work-group tile.
2. **Data staging:** Load A and B tiles from global memory into LDS (shared memory) with coalesced loads. Reuse these tiles across multiple MFMA iterations.
3. **Register packing:** Move sub-blocks from LDS into vector registers arranged to match the MFMA operand layout.
4. **Issue MFMA instructions in a tight inner loop:** For each K-subblock, execute a sequence of MFMA ops that update accumulators.
5. **Post-process:** Apply scaling, add bias, or activation, and write accumulators back to global memory.

This is the same high‑level structure used by tuned GEMM libraries — the difference here is the inner compute is assembly MFMA instead of compiler-generated code.

---

## 🔧 Register and accumulator layout patterns

* **Accumulator organization:** Decide whether accumulators are stored in general-purpose vector registers or a dedicated accumulator register space (depending on ISA). Map each accumulator to a unique register group to avoid write/back hazards.
* **Operand register packing:** Pack multiple elements per vector register to match MFMA lane expectations. For example, if MFMA consumes 4×4 blocks, group 4 elements of a row (or column) into a single vector register.
* **Avoid register spills:** MFMA sequences can consume many registers. Compute the total registers used (operands + accumulators + temporaries) and ensure it stays below the per-thread register file budget to maintain occupancy.

Example (conceptual):

* `vA0, vA1, ...` hold packed rows of A-subtile
* `vB0, vB1, ...` hold packed cols of B-subtile
* `vCacc0, vCacc1, ...` hold accumulators for output micro-tile

---

## 🧾 Example pseudo-assembly (conceptual)

> **Important:** The following is *pseudo-assembly* — instruction names and encodings are illustrative, not exact. Use your ISA manual or the assembler to discover the correct mnemonics for your target GPU.

```
// Assume vA*, vB* are preloaded and packed correctly
// vCacc* are accumulator registers
// K loop unrolled for clarity
for k in 0..(K_tile / mfma_k) {
    MFMA vCacc0, vA0, vB0        // vCacc0 += A_block * B_block
    MFMA vCacc1, vA0, vB1
    MFMA vCacc2, vA1, vB0
    MFMA vCacc3, vA1, vB1
    // ... additional MFMA to cover micro-tile
    // Insert scheduling gaps / independent ops to hide latency
}
```

Key points:

* Issue MFMA instructions that operate on independent accumulator groups so that the scheduler can pipeline them.
* Interleave loads or address calculations between MFMA groups to avoid bottlenecks.

---

## 🖥 Writing assembly: options

1. **Inline assembly in HIP/Clang:** Use inline `asm` blocks or `.inst` directives to emit machine encoding for MFMA. Inline assembly gives the convenience of mixing C++ and assembly but is sensitive to compiler register allocation.

2. **Separate assembly files:** Write `.s` assembly files for the target ISA and assemble with the toolchain. This gives full control and predictable register assignment but is less convenient for integration.

3. **Use LLVM IR or intrinsics:** Some MFMA operations may be exposed as LLVM intrinsics or compiler builtins that generate MFMA; this is a more portable route if supported.

**Inline assembly considerations:**

* Constrain clobbers correctly — you must list v- and s-registers the asm touches.
* Use explicit register naming when possible to control allocation.
* Beware of compiler optimizations that may move loads/stores across your asm blocks — use `volatile` or constraints to avoid reordering.

---

## 🧪 Correctness & validation

* **Unit tests:** Validate microkernels with small matrices and compare against a high‑precision CPU reference implementation.
* **Bit-for-bit checks:** MFMA often uses FMA semantics; be aware of rounding and floating-point differences.
* **Sanity checks:** Verify data packing/unpacking and accumulator layout, as a single lane misplacement can corrupt many outputs.
* **Use disassembly:** Compile and dump GPU code (e.g., `llvm-objdump` / `rocobjdump`) to verify MFMA op encodings and order.

---

## 📈 Performance tuning and pitfalls

* **Occupancy vs registers:** MFMA microkernels use many registers; balance register usage so enough wavefronts remain active to hide MFMA latency.
* **Instruction scheduling:** MFMA latency requires other independent instructions (loads, arithmetic) to be scheduled between MFMA issues — interleave operations to keep pipelines occupied.
* **LDS vs register buffering:** Overusing LDS can reduce occupancy; use registers for per-thread micro-tiles where possible, and LDS to share larger tiles across threads.
* **Memory bandwidth:** Ensure A/B tiles are loaded coalesced and prefetching is used so MFMA is not starvation-limited by reads.
* **Precision modes:** Some MFMA variants support mixed precision (FP16 inputs, FP32 accumulation) — use them where acceptable to increase effective throughput.

Common pitfalls:

* Misaligned or incorrectly packed operands that degrade MFMA efficiency.
* Insufficient independent work between MFMA instructions, causing pipeline stalls.
* Excessive register usage causing spills to memory.

---

## 🔐 Safety & portability

* **Portability:** MFMA opcodes vary between hardware generations. Wrap assembly in device‑generation checks or use intrinsics that the compiler maps to the correct instruction.
* **Fall-back paths:** Provide a portable C/metal fallback (e.g., generic FMA loops) for older GPUs or for debugging.


---
[← Kernel Optimization](kernel_optimization.md)
