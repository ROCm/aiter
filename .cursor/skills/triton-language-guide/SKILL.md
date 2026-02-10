---
name: triton-language-guide
description: Comprehensive Triton GPU programming language guide for writing high-performance kernels on AMD GPUs. Use when writing any Triton kernel code, needing Triton API reference, understanding triton.language operations, or optimizing Triton kernels for AMD Instinct GPUs (MI300X, MI350). Covers the full triton.language API, programming model, AMD-specific optimizations, autotuning, and compilation flow.
---

# Triton Language Guide for AMD GPUs

## What is Triton

Triton is a Python-based domain-specific language for writing GPU kernels. It abstracts away low-level GPU details (thread management, memory coalescing, shared memory) while providing control over tiling and memory access patterns. The Triton compiler translates Python code into optimized AMDGCN assembly via: Python AST -> Triton-IR -> Triton-GPU IR -> LLVM-IR -> AMDGCN -> HSACO binary.

## Programming Model

- A **program** is one instance of a kernel running on one block of threads
- `tl.program_id(axis)` returns the program's index in the grid
- `tl.num_programs(axis)` returns the grid size along an axis
- Programs operate on **blocks** (N-dimensional tensors) not individual elements
- The wavefront size on AMD GPUs is **64** (not 32 like NVIDIA)

### Launching Kernels

```python
import triton

@triton.jit
def my_kernel(X_ptr, Y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask)
    tl.store(Y_ptr + offs, x * 2, mask=mask)

# Launch with grid
grid = (triton.cdiv(N, BLOCK_SIZE),)
my_kernel[grid](x_ptr, y_ptr, N, BLOCK_SIZE=1024)
```

### Autotuning

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},
                       num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32},
                       num_stages=2, num_warps=8),
    ],
    key=["M", "N", "K"],  # Re-tune when these change
)
@triton.jit
def matmul_kernel(..., BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, ...):
    ...
```

### Heuristics

Compute derived constexpr values from runtime arguments:

```python
@triton.heuristics({
    "EVEN_K": lambda args: args["K"] % args["BLOCK_K"] == 0,
})
@triton.jit
def kernel(..., EVEN_K: tl.constexpr):
    if EVEN_K:
        x = tl.load(ptr)  # No mask needed
    else:
        x = tl.load(ptr, mask=mask, other=0.0)
```

## Complete triton.language API Reference

### Tensor Creation
| Function | Description |
|----------|-------------|
| `tl.arange(start, end)` | Contiguous values in `[start, end)` |
| `tl.zeros(shape, dtype)` | Tensor of zeros |
| `tl.full(shape, value, dtype)` | Tensor filled with value |
| `tl.cast(x, dtype)` | Cast tensor to dtype |

### Memory Operations
| Function | Description |
|----------|-------------|
| `tl.load(ptr, mask=None, other=0.0)` | Load from memory. `mask` prevents out-of-bounds. `other` is default value |
| `tl.store(ptr, value, mask=None)` | Store to memory. `mask` prevents out-of-bounds writes |
| `tl.make_block_ptr(base, shape, strides, offsets, block_shape, order)` | Create block pointer for structured access |
| `tl.advance(block_ptr, offsets)` | Advance a block pointer |

**Load/Store parameters:**
- `cache_modifier`: `".ca"` (cache all), `".cg"` (cache global, evict L1), `".cs"` (streaming)
- `eviction_policy`: `"evict_first"`, `"evict_last"`, `"evict_normal"`

### Shape Manipulation
| Function | Description |
|----------|-------------|
| `tl.reshape(x, shape)` | Reshape tensor |
| `tl.expand_dims(x, axis)` | Add dimension |
| `tl.broadcast_to(x, shape)` | Broadcast to shape |
| `tl.trans(x)` | Transpose (aliases `tl.permute`) |
| `tl.join(a, b)` | Join tensors in new minor dimension |
| `tl.split(x)` | Split along last dim (must be size 2) |
| `tl.ravel(x)` | Flatten to 1D |
| `tl.view(x, shape)` | View with different shape |

### Linear Algebra
| Function | Description |
|----------|-------------|
| `tl.dot(a, b, acc=None, input_precision="ieee")` | Matrix multiply. Supports int8, fp8, fp16, bf16, fp32 |
| `tl.dot_scaled(a, b, a_scale, b_scale, ...)` | Scaled matmul (microscaling formats, CDNA4) |

**`tl.dot` key details:**
- Input: 2D or 3D (batched). Inner dims must match.
- `input_precision`: `"ieee"` (exact), `"tf32"` (faster, less precise), `"tf32x3"` (3x tf32)
- `acc`: Optional accumulator tensor for `acc += a @ b`
- Minimum K-dim: 16 for fp16/bf16, 32 for fp8/int8

### Reduction Operations
| Function | Description |
|----------|-------------|
| `tl.sum(x, axis=None)` | Sum reduction |
| `tl.max(x, axis=None)` | Max reduction |
| `tl.min(x, axis=None)` | Min reduction |
| `tl.argmax(x, axis)` | Index of maximum |
| `tl.argmin(x, axis)` | Index of minimum |
| `tl.reduce(input, axis, combine_fn)` | Custom reduction |
| `tl.xor_sum(x, axis)` | XOR reduction |

### Scan/Sort Operations
| Function | Description |
|----------|-------------|
| `tl.cumsum(x, axis)` | Cumulative sum |
| `tl.cumprod(x, axis)` | Cumulative product |
| `tl.associative_scan(input, axis, combine_fn)` | Custom scan |
| `tl.sort(x, dim, descending)` | Sort along dimension |
| `tl.histogram(x, num_bins)` | Histogram with bins of width 1 |
| `tl.gather(x, indices, axis)` | Gather along dimension |

### Math Operations
| Function | Description |
|----------|-------------|
| `tl.abs(x)` | Absolute value |
| `tl.exp(x)`, `tl.exp2(x)` | Exponential (base e, base 2) |
| `tl.log(x)`, `tl.log2(x)` | Logarithm (base e, base 2) |
| `tl.sqrt(x)`, `tl.rsqrt(x)` | Square root, inverse sqrt |
| `tl.sin(x)`, `tl.cos(x)` | Trigonometric |
| `tl.sigmoid(x)` | Sigmoid |
| `tl.erf(x)` | Error function |
| `tl.clamp(x, min, max)` | Clamp to range |
| `tl.maximum(x, y)`, `tl.minimum(x, y)` | Element-wise max/min |
| `tl.cdiv(x, div)` | Ceiling division |
| `tl.fma(x, y, z)` | Fused multiply-add: x*y+z |
| `tl.where(cond, x, y)` | Conditional select |

### Atomic Operations
| Function | Description |
|----------|-------------|
| `tl.atomic_add(ptr, val, mask)` | Atomic add |
| `tl.atomic_max(ptr, val, mask)` | Atomic max |
| `tl.atomic_min(ptr, val, mask)` | Atomic min |
| `tl.atomic_cas(ptr, cmp, val, mask)` | Atomic compare-and-swap |
| `tl.atomic_xchg(ptr, val, mask)` | Atomic exchange |
| `tl.atomic_and/or/xor(ptr, val, mask)` | Atomic bitwise |

### Compiler Hints (Critical for AMD Performance)
| Function | Description |
|----------|-------------|
| `tl.assume(cond)` | Tell compiler condition is true (enables optimizations) |
| `tl.multiple_of(x, values)` | Assert x values are multiples of given values |
| `tl.max_contiguous(x, values)` | Assert first N values are contiguous |
| `tl.max_constancy(x, values)` | Assert first N values are constant |
| `tl.debug_barrier()` | Synchronize all threads in block |

### Data Types
```python
tl.float8e4m3fn   # FP8 (4-bit exponent, 3-bit mantissa) - E4M3
tl.float8e5m2     # FP8 (5-bit exponent, 2-bit mantissa) - E5M2
tl.float16        # FP16
tl.bfloat16       # BF16
tl.float32        # FP32
tl.float64        # FP64
tl.int8           # INT8
tl.int16          # INT16
tl.int32          # INT32
tl.int64          # INT64
tl.uint8/16/32/64 # Unsigned integers
```

### Iteration
```python
# Static range (unrolled at compile time)
for i in tl.static_range(0, 4):
    ...

# Dynamic range with pipelining
for i in tl.range(0, K, BLOCK_K, num_stages=2):
    ...
```

## AMD-Specific Optimization Guide

For detailed AMD GPU optimization, see [amd-optimization.md](amd-optimization.md).

## Online References

- Triton API docs: https://triton-lang.org/main/python-api/triton.language.html
- Triton tutorials: https://triton-lang.org/main/getting-started/tutorials/
- AMD Triton optimization wiki: https://github.com/ROCm/triton/wiki
- AMD Triton blog: https://rocm.blogs.amd.com/software-tools-optimization/kernel-development-optimizations-with-triton-on-/README.html
