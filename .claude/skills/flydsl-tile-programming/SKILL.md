---
name: flydsl-tile-programming
description: >
  Guided step-by-step wizard for producing a new FlyDSL GPU kernel from a requirement:
  classify the kernel type, pick a skeleton, fill in compute, add control flow / sync / LDS,
  then test on GPU. Use when the user wants to WRITE a new kernel, port a Triton kernel to
  FlyDSL, or learn tile programming by following a procedure. For API/layout-algebra lookups,
  per-op reference tables, and troubleshooting, use the flydsl-kernel-authoring skill instead.
allowed-tools: Read Edit Bash Grep Glob Agent
---

# FlyDSL Tile Programming

Guide users through writing GPU kernels using FlyDSL's tile programming model (CuTe-style layout algebra). This skill is a step-by-step wizard that takes a kernel requirement and produces a correct, tested FlyDSL kernel.

**Trigger**: User wants to write a new FlyDSL kernel, port a Triton kernel to FlyDSL, or learn tile programming patterns.

**Prerequisites**: FlyDSL installed. In aiter the FlyDSL kernels are VENDORED under
`aiter/ops/flydsl/kernels/` (buffer_ops, vector, tensor_shim), so import those from
`aiter.ops.flydsl.kernels`, not from `flydsl.expr`. GPU access required for testing
(local target is gfx950 / MI350).

**Scope (read this first)**: This skill is the *procedure* — follow the steps in order to produce a kernel. It is the companion to the **flydsl-kernel-authoring** skill, which is the *reference* (the full layout-algebra API surface, per-op tables, environment variables, and an exhaustive troubleshooting list). When you need to look something up rather than follow a step, go to flydsl-kernel-authoring. When modernizing or cleaning up an *existing* kernel (replacing `ArithValue`, `fx.Index`, raw `llvm.call_intrinsic` / `mlir_*` dialects, `scf.IfOp`, per-tile `*_atom_call`, etc. with the current `fx.*` surface), use the **flydsl-kernel-code-cleanup** skill. This wizard links to both instead of duplicating their tables.

---

## Step 1: Classify the Kernel Type

Ask the user what kind of kernel they need. Map to one of these patterns:

| Pattern | Examples | Key Primitives |
|---------|----------|---------------|
| **Elementwise** | vecadd, scale, relu, abs | `logical_divide` + `copy_atom_call` |
| **Reduction** | sum, max, softmax, layernorm | `buffer_load` + warp shuffle + LDS |
| **Tiled Copy** | transpose, permute, gather | `zipped_divide` + `TiledCopy` |
| **GEMM** | matmul, batched gemm | `TiledMma` + `TiledCopy` + LDS |
| **Fused** | fused attention, GEMM+epilogue | Combine GEMM + elementwise |

---

## Step 2: Generate Kernel Skeleton

Based on the pattern, generate the appropriate skeleton. Every FlyDSL kernel has two parts:

```python
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx

@flyc.kernel
def my_kernel(A: fx.Tensor, B: fx.Tensor, ...):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x
    # ... kernel body ...

@flyc.jit
def my_launch(A: fx.Tensor, B: fx.Tensor, ...,
              stream: fx.Stream = fx.Stream(None)):
    my_kernel(A, B, ...).launch(
        grid=(grid_x, grid_y, grid_z),
        block=(block_x, 1, 1),
        stream=stream
    )
```

### Pattern A: Elementwise Kernel

The simplest pattern. Each thread processes `VEC_WIDTH` elements independently.

**Data flow**: Global -> Register -> Compute -> Register -> Global

```python
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import Vector as Vec

BLOCK_DIM = 256
VEC_WIDTH = 4

@flyc.kernel
def elementwise_kernel(
    A: fx.Tensor,
    Out: fx.Tensor,
    BLOCK_DIM: fx.Constexpr[int],
    VEC_WIDTH: fx.Constexpr[int],
):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x

    # === Step 1: Divide global tensor into block-sized tiles ===
    tile_size = BLOCK_DIM * VEC_WIDTH
    tA = fx.logical_divide(A, fx.make_layout(tile_size, 1))
    tOut = fx.logical_divide(Out, fx.make_layout(tile_size, 1))

    # === Step 2: Select this block's tile ===
    tA = fx.slice(tA, (None, bid))
    tOut = fx.slice(tOut, (None, bid))

    # === Step 3: Divide tile for per-thread vectorized access ===
    tA = fx.logical_divide(tA, fx.make_layout(VEC_WIDTH, 1))
    tOut = fx.logical_divide(tOut, fx.make_layout(VEC_WIDTH, 1))

    # === Step 4: Allocate register and set up copy atom ===
    copy_bits = VEC_WIDTH * 32
    copy_atom = fx.make_copy_atom(fx.UniversalCopy(copy_bits), fx.Float32)
    rA = fx.make_rmem_tensor(VEC_WIDTH, fx.Float32)
    rOut = fx.make_rmem_tensor(VEC_WIDTH, fx.Float32)

    # === Step 5: Load -> Compute -> Store ===
    fx.copy_atom_call(copy_atom, fx.slice(tA, (None, tid)), rA)

    vA = Vec(fx.memref_load_vec(rA))
    # --- YOUR COMPUTE HERE ---
    vOut = vA * vA  # example: square
    # --- END COMPUTE ---
    fx.memref_store_vec(vOut, rOut)

    fx.copy_atom_call(copy_atom, rOut, fx.slice(tOut, (None, tid)))

@flyc.jit
def elementwise_launch(
    A: fx.Tensor, Out: fx.Tensor, N: fx.Int32,
    stream: fx.Stream = fx.Stream(None),
):
    tile_size = BLOCK_DIM * VEC_WIDTH
    grid_x = (N + tile_size - 1) // tile_size
    elementwise_kernel(A, Out, BLOCK_DIM, VEC_WIDTH).launch(
        grid=(grid_x, 1, 1), block=(BLOCK_DIM, 1, 1), stream=stream
    )

# === Test ===
N = 1024
A = torch.randn(N, dtype=torch.float32, device="cuda")
Out = torch.empty(N, dtype=torch.float32, device="cuda")
elementwise_launch(A, Out, N, stream=torch.cuda.Stream())
torch.cuda.synchronize()
assert torch.allclose(Out, A * A, atol=1e-5)
```

### Pattern B: Tiled 2D Copy (Transpose, Gather)

Uses `zipped_divide` + `TiledCopy` for 2D data movement with explicit thread-value mapping.

**Data flow**: Global[M,N] -> Fragment -> Global[M,N] (with layout change)

```python
@flyc.kernel
def tiled_copy_kernel(A: fx.Tensor, B: fx.Tensor):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x

    block_m, block_n = 8, 24
    tile = fx.make_tile([
        fx.make_layout(block_m, 1),
        fx.make_layout(block_n, 1)
    ])

    # Wrap as buffer tensors (AMD buffer descriptors)
    A = fx.rocdl.make_buffer_tensor(A)
    B = fx.rocdl.make_buffer_tensor(B)

    # Divide into tiles, select block's tile
    bA = fx.zipped_divide(A, tile)
    bB = fx.zipped_divide(B, tile)
    bA = fx.slice(bA, (None, bid))
    bB = fx.slice(bB, (None, bid))

    # Thread-value layout: how threads cooperate on the tile
    thr_layout = fx.make_layout((4, 1), (1, 1))   # 4 threads along M
    val_layout = fx.make_layout((1, 8), (1, 1))    # each loads 8 along N
    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
    layout_tv = fx.raked_product(thr_layout, val_layout)
    tile_mn = fx.make_tile(4, 8)

    # Build tiled copy and get thread's partition
    tiled_copy = fx.make_tiled_copy(copy_atom, layout_tv, tile_mn)
    thr_copy = tiled_copy.get_slice(tid)
    src = thr_copy.partition_S(bA)
    dst = thr_copy.partition_D(bB)
    frag = fx.make_fragment_like(src)

    # Copy: global A -> frag -> global B
    fx.copy(copy_atom, src, frag)
    fx.copy(copy_atom, frag, dst)
```

### Pattern C: Tiled MMA (GEMM)

Uses `TiledMma` + `TiledCopy` for matrix multiply with AMD MFMA instructions.

**Data flow**: Global -> (TiledCopy) -> Fragment A,B -> (MFMA) -> Fragment C -> Global

```python
block_m, block_n, block_k = 64, 64, 8

@flyc.kernel
def gemm_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x

    # Define tiles
    tileA = fx.make_tile(block_m, block_k)
    tileB = fx.make_tile(block_n, block_k)
    tileC = fx.make_tile(block_m, block_n)

    # Wrap as buffer tensors
    A = fx.rocdl.make_buffer_tensor(A)
    B = fx.rocdl.make_buffer_tensor(B)
    C = fx.rocdl.make_buffer_tensor(C)

    # Divide and select block's tile
    bA = fx.slice(fx.zipped_divide(A, tileA), (None, bid))
    bB = fx.slice(fx.zipped_divide(B, tileB), (None, bid))
    bC = fx.slice(fx.zipped_divide(C, tileC), (None, bid))

    # === MMA setup ===
    # MFMA(M, N, K, AccType) -- hardware instruction shape
    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 4, fx.Float32))

    # Tile the MMA atom across threads: 2x2 = 4 MMA atoms per warp
    tiled_mma = fx.make_tiled_mma(
        mma_atom,
        fx.make_layout((2, 2, 1), (1, 2, 0))  # (M_rep, N_rep, K_rep)
    )
    thr_mma = tiled_mma.thr_slice(tid)

    # === Copy setup (matched to MMA layout) ===
    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    tiled_copy_A = fx.make_tiled_copy_A(copy_atom, tiled_mma)
    tiled_copy_B = fx.make_tiled_copy_B(copy_atom, tiled_mma)
    tiled_copy_C = fx.make_tiled_copy_C(copy_atom, tiled_mma)

    thr_copy_A = tiled_copy_A.get_slice(tid)
    thr_copy_B = tiled_copy_B.get_slice(tid)
    thr_copy_C = tiled_copy_C.get_slice(tid)

    # === Partition data ===
    # Copy partitions (for data movement)
    copy_src_A = thr_copy_A.partition_S(bA)
    copy_src_B = thr_copy_B.partition_S(bB)
    copy_dst_C = thr_copy_C.partition_S(bC)

    # MMA partitions (for compute)
    part_A = thr_mma.partition_A(bA)
    part_B = thr_mma.partition_B(bB)
    part_C = thr_mma.partition_C(bC)

    # === Allocate fragments (registers) ===
    frag_A = thr_mma.make_fragment_A(part_A)
    frag_B = thr_mma.make_fragment_B(part_B)
    frag_C = thr_mma.make_fragment_C(part_C)

    # Retile fragments for copy compatibility
    copy_frag_A = thr_copy_A.retile(frag_A)
    copy_frag_B = thr_copy_B.retile(frag_B)
    copy_frag_C = thr_copy_C.retile(frag_C)

    # === Execute: Load A,B -> GEMM -> Store C ===
    fx.copy(copy_atom, copy_src_A, copy_frag_A, pred=None)
    fx.copy(copy_atom, copy_src_B, copy_frag_B, pred=None)
    fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)
    fx.copy(copy_atom, copy_frag_C, copy_dst_C, pred=None)
```

### Pattern D: Buffer Load/Store (Low-level)

Direct AMD buffer intrinsics for maximum control. Bypasses the layout algebra.

```python
# aiter vendors buffer_ops (flydsl moved it out of flydsl.expr):
from aiter.ops.flydsl.kernels import buffer_ops

@flyc.kernel
def buffer_kernel(A: fx.Tensor, B: fx.Tensor, N: fx.Constexpr[int]):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x
    gid = bid * 256 + tid

    rsrc_a = buffer_ops.create_buffer_resource(A)
    rsrc_b = buffer_ops.create_buffer_resource(B)

    # offset is in ELEMENTS (not bytes!) -- buffer_load converts internally
    data = buffer_ops.buffer_load(rsrc_a, gid * 4, vec_width=4, dtype=fx.T.f32())
    # ... compute on data ...
    buffer_ops.buffer_store(data, rsrc_b, gid * 4)
```

---

## Step 3: Fill in the Compute Logic

Common compute recipes (all work on vectors):

```python
from flydsl.expr.typing import Vector as Vec

# Scale: C = A * scalar
scale = Vec.filled(VEC_WIDTH, 2.0, fx.Float32)
vC = Vec(vA) * scale

# Add: C = A + B
vC = Vec(vA) + Vec(vB)

# FMA: D = A * B + C
vC = Vec(vA) * Vec(vB) + Vec(vC)

# ReLU: C = max(A, 0)
zero = Vec.filled(VEC_WIDTH, 0.0, fx.Float32)
vC = Vec(vA).maximumf(zero)

# Abs: C = |A|
v = Vec(vA)
neg = -v
is_neg = v < zero
vC = is_neg.select(neg, v)

# Type conversion
vC = Vec(vI32).to(fx.Float32)  # int -> float
vC = Vec(vF32).to(fx.Float16)  # f32 -> f16
```

---

## Step 4: Add Control Flow

```python
from flydsl.expr import range_constexpr

# Compile-time unrolled loop (constant bounds)
for i in range_constexpr(K):
    ...

# Runtime loop (dynamic bounds)
for i in range(runtime_N):
    ...

# Loop with carried state (software pipelining)
# Bounds must be typed DSL integers (fx.Int64), NOT plain Python ints — a plain
# int makes the AST rewriter unroll the loop and silently ignore init=.
start, stop, step = fx.Int64(0), fx.Int64(N - 1), fx.Int64(1)
for iv, state in range(start, stop, step, init=[acc_init, ...]):
    acc = state[0]
    # ... compute ...
    results = yield [new_acc, ...]
final_acc = results[0]

# Static if (compile-time, no MLIR)
from flydsl.expr import const_expr
if const_expr(USE_FAST_PATH):
    ...

# Dynamic if (runtime, rewritten by the frontend)
if bid == 0:
    ...
```

---

## Step 5: Add Synchronization (if needed)

```python
# Workgroup barrier (__syncthreads)
fx.gpu.barrier()

# Fine-grained waitcnt (CDNA3)
fx.rocdl.s_waitcnt(0)

# Fine-grained waitcnt (CDNA4 / gfx950)
fx.rocdl.s_wait_loadcnt(0)
fx.rocdl.s_wait_storecnt(0)
fx.rocdl.s_wait_dscnt(0)

# Scheduling hints
fx.rocdl.sched_mfma(N)     # schedule N MFMA before next barrier
fx.rocdl.sched_vmem(N)     # schedule N VMEM reads
fx.rocdl.sched_dsrd(N)     # schedule N DS reads
fx.rocdl.sched_dswr(N)     # schedule N DS writes
```

---

## Step 6: Add Shared Memory (if needed)

Declare the LDS storage as an ``@fx.struct`` of ``fx.Array`` fields and allocate
it inside the kernel with ``fx.SharedAllocator`` (the current LDS API — the
compiler sizes the static LDS global, so ``launch(smem=...)`` is left unset):

```python
@fx.struct
class SharedStorage:
    buf: fx.Array[fx.Float16, num_elements]

@flyc.kernel
def kernel_with_lds(A: fx.Tensor, ...):
    lds = fx.SharedAllocator().allocate(SharedStorage).peek()
    lds_buf = lds.buf.view(fx.make_layout((rows, cols), (cols, 1)))

    # Write to LDS, sync, read back (through view loads/stores or copy atoms)
    fx.gpu.barrier()
```

(The legacy `flydsl.utils.smem_allocator.SmemAllocator` path still exists for
un-migrated kernels but is not recommended for new code.)

LDS capacity: gfx942 (MI300X) = 64KB, gfx950 (MI350) = 160KB.

---

## Step 7: Test the Kernel

Run the kernel locally or on a remote GPU:

```bash
# Run a kernel / its op test (aiter, local gfx950 via /opt/venv)
python op_tests/flydsl_tests/test_<kernel>.py        # script-style tests
python -m pytest op_tests/flydsl_tests/test_<kernel>.py -q

# Force a COLD recompile when verifying a change (bypass the flydsl cache):
FLYDSL_RUNTIME_ENABLE_CACHE=0 rm -rf ~/.flydsl/cache && python ...

# Dump IR for debugging
FLYDSL_DUMP_IR=1 python op_tests/flydsl_tests/test_<kernel>.py
```

Real aiter reference kernels to copy patterns from: `silu_and_mul_fq.py`
(elementwise + quant), `qk_norm_rope_quant.py` / `pa_mqa_logits_fp4.py`
(reduction + MFMA + `make_buffer_tensor`), `mxfp4_gemm1.py` (GEMM), all under
`aiter/ops/flydsl/kernels/`.

---

## Step 8: Debug Common Errors

If the kernel fails to compile or produces wrong results, consult the full error -> cause -> fix
table in the **flydsl-kernel-authoring** skill (§10 Troubleshooting), which covers the common
wizard pitfalls: Python `int` where a DSL value is expected, `NameError` inside extracted
`__then_*` branches, missing `arith.absf`, scalar/vector mismatches, LDS overflow,
`buffer_load` element-vs-byte offsets, `range(..., init=...)` being unrolled, and stale caches.
For deeper kernel-debugging methodology (all-1s test, single-partition isolation, MFMA operand
layout checks), use the **debug-flydsl-kernel** skill.

---

## Tile Programming Mental Model

```
                  Layout Algebra
                  =============
   make_layout(shape, stride)  ->  Layout = mapping: coord -> index

                  Divide (Partition)
                  =================
   zipped_divide(Tensor, Tile)  ->  (tile_interior, tile_id)
   slice(divided, (None, bid))  ->  this block's tile

                  Atom (Hardware Instruction)
                  ==========================
   CopyAtom  = one hardware copy instruction (32b/64b/128b buffer copy;
               or a gfx1250 TDM whole-tile async Global<->LDS DMA)
   MmaAtom   = one matrix instruction (CDNA MFMA 16x16x4/16x16x16/...;
               gfx1250 WMMA / MX-scaled WMMA, wave32)

                  Tiled Operation (Thread Cooperation)
                  ====================================
   TiledCopy = CopyAtom x thread_layout  -> many threads cooperate on copy
   TiledMma  = MmaAtom  x atom_layout   -> many threads cooperate on MMA

                  Per-Thread View
                  ===============
   ThrCopy.partition_S/D(tensor)  ->  this thread's source/dest data
   ThrMma.partition_A/B/C(tensor) ->  this thread's operand data

                  Fragment (Register Storage)
                  ==========================
   make_fragment_like(partition)  ->  register tile
   retile(fragment)              ->  reshape for copy compatibility

                  Execute
                  =======
   fx.copy(atom, src, dst)           ->  data movement
   fx.gemm(atom, D, A, B, C)        ->  matrix multiply: D = A @ B + C
```

Key insight: **Layout is the glue**. Every operation (divide, partition, copy, gemm) is defined in terms of layouts that describe the mapping from logical coordinates to physical locations. Getting the layouts right is 90% of FlyDSL programming.

---

## MFMA Instruction Reference (AMD CDNA3/4)

For the table of available MFMA instruction shapes (`MFMA(16,16,4,Float32)`,
`MFMA(16,16,16,Float32)`, FP8/BF16/CDNA4-scaled variants), the **gfx1250 WMMA /
MX-scaled WMMA** atoms (wave32, `rocdl.WMMA` / `rocdl.WMMAScale`), and how
`make_tiled_mma`'s `(M_rep, N_rep, K_rep)` atom_layout works, see the
**flydsl-kernel-authoring** skill (§6 MFMA Integration + gfx1250 WMMA). Use it when
choosing the MMA atom for the Pattern C (GEMM) skeleton above.

---

## Checklist for New Kernels

- [ ] Identified kernel pattern (elementwise / reduction / copy / GEMM)
- [ ] Chose appropriate copy atom type (Universal vs Buffer, bit width)
- [ ] Set tile sizes matching MFMA instruction shape (if GEMM)
- [ ] Verified VEC_WIDTH * sizeof(elem) <= copy atom bits
- [ ] Used `Constexpr[int]` for compile-time constants, `Int32` for runtime
- [ ] Added `torch.cuda.synchronize()` before checking results
- [ ] Verified correctness with `torch.allclose()`
