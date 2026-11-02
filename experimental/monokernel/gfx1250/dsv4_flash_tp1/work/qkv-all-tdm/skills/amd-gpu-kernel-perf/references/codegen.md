# Codegen traps: Gluon / Triton, FlyDSL, LLVM, libraries — and how to see the real ISA

Versions: observed with Triton 3.8.0 Gluon (stock vLLM ROCm image), FlyDSL 0.3.2-0.4.0, AITER of the same image,
gfx950. Re-check on newer stacks; several of these are compiler bugs that may get fixed.

## Contents
1. Inspecting ISA (do this first)
2. Loads and stores
3. Registers, spills and inlining
4. LDS layouts and MFMA
5. Inline asm hazards
6. Frontend restrictions (Gluon)
7. LLVM scheduler behavior in GEMMs
8. Libraries: AITER, FlyDSL, hipBLASLt
9. Compile time and caches

## 1. Inspecting ISA

- The kernel that actually launched is the ground truth. Dump its `.s` / `.ll` from the runtime path (an env var
  hook that writes `kernel.asm["amdgcn"]` and `["llir"]`), because `--rm` containers keep no Triton cache.
- GPU-free compile probe: build with `GluonASTSource` + `triton.compile(..., target=GPUTarget("hip", "gfx950", 64))`
  and report VGPRs, spills, LDS. Pass `tt.divisibility=16` for pointer arguments or the probe shows fake byte-split
  loads that the real launch does not have. `kernel.warmup(*args, grid=...)` compiles without launching (also the
  way to precompile inside a serving warm-up).
- Count in the asm: `global_load_*` / `buffer_load_*` widths (dwordx4 is what you want), `s_waitcnt vmcnt(0)`
  inside hot loops, `scratch_` instructions (spills), `s_barrier`, `v_mfma*`. The ISA carries source-line info; use
  it to attribute spills to lines.
- Beware dump artifacts: two code paths interleaved in the listing once made it look like no loads were issued
  before a poll.
- To prove two runs used identical kernels, compare the compiled code objects, not kernel names: library JIT caches
  (FlyDSL `.pkl` artifacts embed a `gpu.binary` code object) can be byte-compared and disassembled with
  `llvm-objdump -d` (ROCm's llvm, e.g. `_rocm_sdk_core/lib/llvm/bin/llvm-objdump` in pip-ROCm images). One cache can
  hold two different binaries with the same kernel name (different compile keys), so also check which key a call
  selects (fresh cache per configuration, list what gets compiled).

## 2. Loads and stores

- `gl.load(..., cache_modifier=".cv")` lowers to one scalar load per element, each followed by `s_waitcnt
  vmcnt(0)`. Use `gl.amd.cdna4.buffer_load(..., cache=".cv")` on int32/int64 and reinterpret.
- u8 / bf16 `buffer_load` never vectorizes; plain `gl.load` of u8 weights gives `global_load_dwordx4`.
- Masked global loads become scalar loads plus exec branches (2192 spills in one case). Prefer unmasked loads
  under a uniform `if live:` branch, or clamp row indices instead of masking.
- Missing `gl.multiple_of` / divisibility on strides or pointers scalarizes loads into `global_load_ushort`.
- Loop-carried u8 tensors get one VGPR per byte (613-873 spills); unroll statically.
- buffer_load offsets must be int32.
- Gluon on the stock image lacks `gl.atomic_load` / `gl.atomic_store`; relaxed atomic stores lower to plain
  coherent stores, so `.wt` stores and `.cv` buffer loads (or a small shim) are equivalent.

## 3. Registers, spills and inlining

- A persistent kernel must keep every CTA resident; treat any VGPR spill as a correctness risk, not just a slowdown
  (a spill-heavy 8-row build faulted). AITER's host refuses to launch kernels with spills. Target zero.
- Fully inlining all phases of a layer into one function shares one register budget across all of them (206-438
  spills). Options: `noinline` phase wrappers (each phase ~140 VGPRs; costs ~3 us per hop for callee-saved
  save/restore and `s_waitcnt` at return), or an inlined stage walk with per-row-count specializations.
  Module-wide inlining leaks into other entry points: wrap the other entry's phases in `noinline`.
- Packing ~70 pointer arguments into one weight/scratch blob with constexpr offsets removed 279 spills.
- A runtime multi-way branch that a constexpr could decide spilled; specialize at compile time.
- Holding a deep prefetch in registers costs VGPRs: moving a wait after the loads cost 7 spills. Use async copies
  into LDS for depth beyond the register budget (raised in-flight bytes from ~256 KB to 320 KB per CTA).

## 4. LDS layouts and MFMA

- Unswizzled LDS rows at power-of-two strides (2048 B, 512 B) give 16-way bank conflicts. Use swizzled per-chunk
  `[chunks, rows, cols]` buffers (`SwizzledSharedLayout(...)`).
- Shared-memory `.index()` needs a layout of rank n-1 (use a `_reinterpret` view); whole-tensor stores need the full
  rank. Slicing along a swizzled dim is unsupported; memdesc `.slice` offsets must be tile-aligned.
  `allocate_shared_memory([2,2,32,64])` with a 2-D layout fails ("rank must be equal to or one less").
- A 5-D `_reinterpret(...).permute(...)` view of a pre-shuffled LDS image compiles to a free layout conversion;
  a naive reshape that "looks register-identical" was numerically wrong. Validate layout tricks numerically.
- `gl.permute` of a dot-operand layout into the other operand's layout is register-identical (free); `gl.join`
  caused LDS round trips.
- Scaled MFMA (A4W4, `mfma_scaled` 16x16x128 e2m1) works on 2-D operands only; batched rank-3 fails in lowering.
  Stack along M instead.

## 5. Inline asm hazards

- An inline asm that reads an MFMA result needs explicit wait states (`s_nop 7` x3 before it); LLVM inserts none for
  asm. Without them results were nondeterministically wrong in some columns — the biggest single bug of one project.
  Only the asm before a real read needs them.
- For pure ordering use an empty asm whose output is tied to its input (`"=v,0,~{memory}"`).
- Inline asm needs explicit input constraints; outputs of inactive lanes are undefined.
- `v_dot2_f32_bf16` inline asm produced wrong results whenever async LDS copies were in the kernel; abandoned.
- Wrappers for exact reference rounding: `v_mul_f32` / `v_rndne_f32` / `v_ldexp_f32`.

## 6. Frontend restrictions (Gluon, Triton 3.8)

- Module globals must be `gl.constexpr(...)`; no nested defs; no Python lists; `static_range` bounds constexpr; a
  constexpr defined inside an `if` is unsupported; tensor dims are powers of two (mask a 14 up to 16).
- `constexpr_function` cannot call plain Python functions (precompute a dict).
- A constexpr-multiply form (`(S == 1) * ticks`) folded to 0; write an explicit if/else.
- `gl.broadcast_to` does not exist (expand_dims + add zeros); `x[None, None, :]` needs
  `SliceLayout(0, SliceLayout(1, L))`; scans on slice layouts are unsupported (masked in-thread sum).
- A runtime ring index failed to lower; use `static_range`.
- Mark per-call ints (layer, tag, rows) `do_not_specialize`, or each value compiles a variant. Warm-up must pass
  stand-ins of the exact dtypes: an fp32 tensor where bf16 is expected compiles a different variant (use
  `.view(dtype)` reinterpretations of a scratch buffer).
- `TRITON_FRONT_END_DEBUGGING=1` gives full tracebacks.

## 7. LLVM scheduler behavior in GEMMs

- The scheduler collapses register double-buffering: it reloads each MFMA's operands from LDS right before use and
  waits, exposing LDS latency. Register-target and scheduling-strategy knobs did not fix it; wider in-flight loads
  spilled the accumulator. Untested next steps were 8 waves (two per SIMD) and Gluon `warp_pipeline`.
- Result in practice: a hand-written Gluon GEMM at M=240 reached ~16 us where hipBLASLt took ~13.5 us; a Triton
  config sweep found nothing better than the serving framework's GEMMs. Prefer library GEMMs at large M.

## 8. Libraries

- AITER FlyDSL kernels (MoE stage 1/2, hgemm) are bound to the first device a process touches ("invalid device
  ordinal" elsewhere): multi-GPU tests need one process per GPU.
- Find which library kernels run by static analysis of the call site and the tuned config tables before spending GPU
  time (e.g. AITER fused MoE picks kernels from a tuned CSV keyed on M padded to the next power of two; 240 rows ->
  the 256 row: sort, 2-kernel quant, stage-1 GEMM, stage-2 GEMM with atomic accumulate into a zeroed buffer).
  A neighboring row of the table may be faster (the 512 row beat the 256 row): a retuning lead.
- `torch.matmul(..., out=)` may select a different backend (hipBLASLt) than the framework's linear layer (Triton
  split-K): o_proj 14.4 vs 16.8 us, a K=768 GEMM 10.4 vs 15.1 us at M=240.
- `.zero_()` fails on fp4 dtypes; `.view(torch.uint8).zero_()`.
- A library call can launch several kernels (sort, quant, zeroing, the GEMMs); read the dispatch, then confirm with a
  trace.

## 9. Compile time and caches

- Compile in warm-up, never inside CUDA-graph capture. A cold Gluon JIT made a server start take 610 s vs 260 s.
- Keep a persistent compile cache directory per configuration; for A/B, separate caches let you list exactly which
  kernels each arm compiled.
