# Opus Guide

Opus (**AI (O)(P)erator Micro(U) (S)TD**) is a lightweight, single-header C++ template library for writing HIP kernels on AMD GPUs. Inspired by [ck_tile](https://github.com/ROCm/composable_kernel) and [cutlass/cute](https://github.com/NVIDIA/cutlass), opus provides a simplified and maintainable set of abstractions for data types, vectorized memory access, layout descriptors, and matrix core (MFMA) operations.

## Table of Contents

- [Quick Reference](#quick-reference)
- [1. Design Philosophy](#1-design-philosophy)
- [2. Type System](#2-type-system)
- [3. Compile-Time Constants](#3-compile-time-constants)
- [4. Container Types](#4-container-types)
- [5. Layout System](#5-layout-system)
- [6. Global Memory Access (gmem)](#6-global-memory-access-gmem)
- [7. Shared Memory Access (smem)](#7-shared-memory-access-smem)
- [8. Type Casting and Conversion](#8-type-casting-and-conversion)
- [9. MFMA (Matrix Core) Operations](#9-mfma-matrix-core-operations)
- [10. Tiled MMA (Multi-Wave GEMM)](#10-tiled-mma-multi-wave-gemm)
- [11. Distributed Tensor Views (x/p/y dims)](#11-distributed-tensor-views-xpy-dims)
- [12. Utility Functions](#12-utility-functions)
- [13. Usage in AITER Kernels](#13-usage-in-aiter-kernels)
- [Decision Tree: When to Use Opus](#decision-tree-when-to-use-opus)
- [Source Files](#source-files)

## Quick Reference

```cpp
#include "opus/opus.hpp"

// Enable opus types and literals inside kernel
OPUS_USING_COMMON_TYPES

using namespace opus;

// Data types
fp16_t val;                         // Half precision float
bf16_t bval;                        // BFloat16
fp8_t  f8val;                       // FP8 (E4M3)
fp4_t  f4val;                       // FP4 (packed, 2 values per byte)
fp16x4_t vec4;                      // Vector of 4 fp16 values

// Compile-time constants
auto n = 42_I;                      // number<42>
auto seq_vals = seq<2, 4, 8>{};     // Compile-time integer sequence

// Containers
auto t = make_tuple(1_I, 2_I, 3_I); // Heterogeneous tuple
auto a = make_array(1.0f, 2.0f);    // Homogeneous array
auto v = make_vector(1.0f, 2.0f);   // GPU vector (ext_vector_type)

// Layout for index calculation
auto u = make_layout(make_tuple(128_I, 64_I));
int offset = u(4, 8);              // = 4 * 64 + 8 * 1

// Global memory: vectorized load/store with OOB protection
auto g = make_gmem(ptr, num_elements * sizeof(fp16_t));
auto data = g.load<4>(row_offset);  // Load 4 elements as fp16x4_t
g.store<4>(data, row_offset);       // Store 4 elements

// Shared memory
__shared__ fp16_t smem[1024];
auto smem_view = make_smem(smem);
auto smem_val = smem_view.load<4>(offset);

// MFMA: warp-level matrix multiply
auto mma = make_mfma<fp16_t, fp16_t, fp32_t>(32_I, 32_I, 8_I);
auto c = mma(a_reg, b_reg, c_reg);

// Tiled MMA: multi-wave block-level GEMM
auto tmma = make_tiled_mma<fp16_t, fp16_t, fp32_t>(
    seq<2, 1, 1>{},    // expand: repeat 2x along M
    seq<2, 2, 1>{},    // tile: 2x2 wave grid
    seq<16, 16, 16>{},  // wave MNK
    mfma_adaptor_swap_ab{}
);
```

---

## 1. Design Philosophy

Opus occupies a specific niche in the GPU kernel development stack:

```
Hand-written HIP kernels  <  opus  <  ck_tile / cutlass
(low-level, error-prone)    (sweet spot)   (full-featured, complex)
```

**What opus provides:**
- AMDGPU data type declarations and conversions (FP16, BF16, FP8, FP4, INT8, etc.)
- Automated vectorized buffer load/store dispatch
- Support for multiple MFMA instruction types with minimal code changes
- Utility device functions (DPP shuffles, reductions, waitcnt)
- Simple layout abstractions for index calculations

**What opus does NOT provide:**
- Pre-optimized kernels (GEMM, attention, reduction)
- Reusable device-side pipelines
- Comprehensive layout transformation algebra (use ck_tile or cutlass for that)

The library is distributed as a **single header file** (`opus.hpp`, ~1813 lines), split into two logical sections:
1. **Device-independent**: containers (`seq`, `tuple`, `array`, `vector`), `layout`, functional utilities
2. **Architecture-specific**: data types, buffer load/store, MFMA instructions, DPP operations

---

## 2. Type System

### Scalar Types

Opus registers GPU-compatible scalar types with the `REGISTER_DTYPE` macro, providing each type along with vector variants (x1 through x64):

| Type | Underlying | Description |
|------|-----------|-------------|
| `fp32_t` | `float` | 32-bit float |
| `fp16_t` | `_Float16` / `__fp16` | 16-bit float (ROCm version dependent) |
| `bf16_t` | `unsigned short` / `__bf16` | BFloat16 (ROCm version dependent) |
| `fp8_t` | `_BitInt(8)` | FP8 E4M3 |
| `bf8_t` | `unsigned _BitInt(8)` | BF8 E5M2 |
| `i32_t` / `u32_t` | `int32_t` / `uint32_t` | 32-bit integers |
| `i16_t` | `int16_t` | 16-bit signed integer |
| `u16_t`* | `uint16_t` | 16-bit unsigned integer (*requires `__clang_major__ >= 20`, i.e. ROCm 7+*) |
| `i8_t` / `u8_t` | `int8_t` / `uint8_t` | 8-bit integers |

### Vector Types

Every registered scalar type automatically gets vector variants:

```cpp
fp16x4_t  vec4;   // 4x fp16
fp32x8_t  vec8;   // 8x fp32
bf16x16_t vec16;  // 16x bf16
```

Pattern: `<type>x<N>_t` where N = 1, 2, 4, 8, 16, 32, 64.

### Sub-byte Packed Types

For sub-byte data types (FP4, INT4), opus uses a packed representation where multiple values share a byte:

| Type | Storage | Bits | Description |
|------|---------|------|-------------|
| `fp4_t` | `uint8_t` | 4 (E2M1) | FP4, packed 2 per byte |
| `int4_t` | `uint8_t` | 4 | Signed INT4, packed 2 per byte |
| `uint4_t` | `uint8_t` | 4 | Unsigned INT4, packed 2 per byte |
| `e8m0_t` | `uint8_t` | 8 (E8M0) | MX scaling format |

### Type Traits

```cpp
is_dtype_v<T>     // True if T is a registered opus data type
is_vector_v<T>    // True if T is an ext_vector_type
is_packs_v<T>     // True if T is a sub-byte packed type
sizeof_bits_v<T>  // Size in bits (works for sub-byte types)
num_packs_v<T>    // Number of values packed in one storage unit
```

---

## 3. Compile-Time Constants

### `number<I>` — Static Integer

A compile-time integer constant. Operations on two `number<>` values produce another `number<>`:

```cpp
auto a = number<4>{};
auto b = number<8>{};
auto c = a + b;            // number<12>
auto d = a * b;            // number<32>
auto e = b % a;            // number<0>
```

### `_I` Literal Suffix

Convenient shorthand for creating `number<>` values:

```cpp
auto n = 42_I;             // number<42>
auto m = 16_I;             // number<16>

// Enable the literal in your kernel scope:
using opus::operator""_I;
// Or use OPUS_USING_COMMON_TYPES macro
```

### `seq<Is...>` — Compile-Time Integer Sequence

A compile-time sequence of integers, useful for describing shapes, tile configurations, and loop bounds:

```cpp
auto s = seq<2, 4, 8>{};
s[0];                       // 2
s.size();                   // 3

// Creation helpers
make_index_seq<5>           // seq<0,1,2,3,4>
make_index_seq<2, 7>        // seq<2,3,4,5,6>
make_index_seq<0, 10, 2>    // seq<0,2,4,6,8>
make_repeated_seq<0, 4>     // seq<0,0,0,0>

// Operations
concat_seq(seq<1,2>{}, seq<3,4>{})  // seq<1,2,3,4>
reduce_seq_sum(seq<1,2,3>{})        // seq<6>
reduce_seq_mul(seq<2,3,4>{})        // seq<24>
```

---

## 4. Container Types

### `tuple<T...>` — Heterogeneous Container

Stores elements of different types. Supports structured bindings:

```cpp
auto t = make_tuple(1_I, 2.0f, 3_I);
auto v0 = get<0>(t);                    // number<1>
auto v1 = get<1>(t);                    // 2.0f
t.size();                               // 3

// Structured binding
auto [a, b, c] = make_tuple(1, 2, 3);

// Operations
concat_tuple(t1, t2)                    // Concatenate
flatten_tuple(nested_tuple)             // Flatten nested tuples
reduce_tuple_sum(t)                     // Sum all elements
reduce_tuple_mul(t)                     // Multiply all elements
transform_tuple(f, t)                   // Apply f to each element
make_repeated_tuple<N>(val)             // N copies of val
```

### `array<T, N>` — Homogeneous Container

Fixed-size array of same-type elements:

```cpp
auto a = make_array(1.0f, 2.0f, 3.0f);
a[0];                                    // 1.0f
a.fill(0.0f);                           // Set all to 0
a.clear();                              // Set all to 0

// Operations
concat_array(a1, a2)                    // Concatenate
to_vector(a)                            // Convert to vector type
```

### `vector_t<T, N>` — GPU Vector

Wrapper for `__attribute__((ext_vector_type(N)))`, the native GPU SIMD type:

```cpp
vector_t<float, 4> v;                   // Same as fp32x4_t
auto v = make_vector(1.0f, 2.0f);       // fp32x2_t
v[0] = 3.0f;                            // Element access

// Operations
concat_vector(v1, v2)                   // Concatenate vectors
make_repeated_vector<4>(0.0f)           // 4 zeros
fill(v, 1.0f)                           // Set all elements
clear(v)                                // Zero out

// Conversions
to_array(v)                             // vector -> array
to_vector(a)                            // array -> vector
```

### Slicing

All containers support static slicing:

```cpp
auto v = make_vector(1.0f, 2.0f, 3.0f, 4.0f);
auto s = slice(v, 2_I);                 // First 2 elements
auto s = slice(v, 1_I, 3_I);            // Elements [1, 3)
set_slice(dst, src, 2_I, 4_I);          // Write src into dst[2:4]
```

---

## 5. Layout System

The `layout` descriptor computes linear offsets for N-dimensional tensors. It stores a shape, stride, and optional coordinate:

```
offset = index[0] * stride[0] + index[1] * stride[1] + ...
```

### Creating Layouts

```cpp
// Packed layout (strides computed automatically as row-major)
auto u = make_layout(make_tuple(128_I, 64_I));
// shape = (128, 64), stride = (64, 1)
u(4, 8);                                // = 4 * 64 + 8 = 264

// Explicit strides
auto u = make_layout(
    make_tuple(128_I, 64_I),             // shape
    make_tuple(64_I, 1_I)                // stride
);

// Shape-only (scalar args -> auto-packed)
auto u = make_layout(128_I, 64_I);       // Same as above

// Dynamic strides
auto u = make_layout(
    make_tuple(128, 64),                  // shape (dynamic)
    make_tuple(stride_m, 1)               // stride (dynamic)
);
```

### Layout Variants

The `cached_vec` template parameter controls the layout implementation:

| `cached_vec` | Type | Description |
|-------------|------|-------------|
| `< 0` | `layout` | Basic: computes offset on every call |
| `== 0` (default) | `layout_linear` | Adds a mutable `linear_offset` for incremental addressing |
| `> 0` | `layout_cached` | Pre-computes all offsets into an array for fast lookup |

```cpp
// layout_linear: supports += for sliding window patterns
auto u = make_layout<0>(shape, stride);
u += row_stride;                         // Advance by one row

// layout_cached: pre-computed offsets for repeated access patterns
auto u = make_layout<4>(shape, stride);  // cached_vec=4
```

### Query Operations

```cpp
u.shape()           // Get full shape tuple
u.stride()          // Get full stride tuple
u.shape<0>()        // Get first dimension size
u.stride<1>()       // Get second dimension stride
```

---

## 6. Global Memory Access (gmem)

The `gmem` struct wraps AMD's buffer resource descriptor (`__amdgpu_buffer_rsrc_t`) for vectorized global memory operations with optional out-of-bounds (OOB) protection.

### Creating gmem

```cpp
// Without OOB protection (size = 0xffffffff)
auto g = opus::make_gmem(reinterpret_cast<const fp16_t*>(ptr));

// With OOB protection (loads return 0 for out-of-bounds accesses)
auto g = opus::make_gmem(reinterpret_cast<const fp16_t*>(ptr),
                          num_elements * sizeof(fp16_t));
```

### Scalar Load/Store

```cpp
// Load: offset in units of the base type
auto val = g.load(offset);              // Load 1 element
auto vec = g.load<4>(offset);           // Load 4 elements as vector
auto vec = g.load<8>(offset);           // Load 8 elements as vector

// Store
g.store(val, offset);                   // Store 1 element
g.store<4>(vec4, offset);               // Store 4-element vector
```

### Layout-based Bulk Load/Store

When a layout is provided, gmem automatically issues multiple load/store instructions based on the y-dimension (per-thread) shape:

```cpp
auto u = mma.layout_a(make_tuple(stride_m, 1_I), p_coord);
auto tile_a = g.load<4>(u);             // Bulk load entire A tile
g.store<4>(tile_c, u_out);              // Bulk store entire tile
```

### Async Load (Global -> LDS)

Direct global-to-LDS load without going through registers:

```cpp
__shared__ fp16_t smem[1024];
g.async_load<4>(smem, v_offset, s_offset);

// Layout-based async load
g.async_load<4>(smem, u_gmem, u_smem);
```

### The `aux` Template Parameter

The `aux` parameter maps to the GLC/SLC/DLC cache hint bits in buffer instructions:

| Value | Meaning |
|-------|---------|
| 0 | Default (temporal) load/store behavior |
| 3 | Group non-temporal — hints that data won't be reused |

> **Note:** Named constants `aiter::RT` (0) and `aiter::GROUP_NT` (3) are defined in `csrc/include/aiter_opus_plus.h`, not in `opus.hpp` itself.

```cpp
auto val = g.load<4, 3>(offset);         // Non-temporal load hint
```

---

## 7. Shared Memory Access (smem)

The `smem` struct provides vectorized shared memory (LDS) access:

```cpp
__shared__ fp16_t lds[2048];
auto s = opus::make_smem(lds);

// Load/Store (offset in units of base type)
auto vec = s.load<4>(offset);            // Load 4 elements
s.store<4>(vec, offset);                 // Store 4 elements

// Layout-based bulk operations
auto tile = s.load<4>(u_layout);
s.store<4>(data, u_layout);
```

---

## 8. Type Casting and Conversion

### Scalar Conversions

```cpp
auto f32 = opus::fp16_to_fp32(f16_val);
auto f16 = opus::fp32_to_fp16(f32_val);
auto f32 = opus::bf16_to_fp32(bf16_val);
auto bf16 = opus::fp32_to_bf16(f32_val);  // Rounding mode configurable
```

### BF16 Rounding Modes

`fp32_to_bf16` accepts a compile-time rounding mode parameter:

| Mode | Description |
|------|-------------|
| 0 | Standard (round-to-nearest-even) |
| 1 | Truncate with NaN preservation |
| 2 | Truncate (default, `OPUS_FP32_to_BF16_DEFAULT`) |
| 3 | Standard via inline assembly |

On GFX950 (MI350), the compiler uses native conversion instructions regardless of mode.

### Generic Cast Interface

The unified `cast<D>(src)` function dispatches to the correct conversion:

```cpp
auto f32 = opus::cast<fp32_t>(f16_val);   // fp16 -> fp32
auto bf16 = opus::cast<bf16_t>(f32_val);  // fp32 -> bf16
```

### Vectorized Cast

Cast works on vectors, tuples, and arrays — it automatically handles packed conversions:

```cpp
// FP32 -> FP8: packs 2 or 4 values using hardware instructions
auto f8_packed = opus::cast<fp8_t>(fp32x4_val);

// FP8 -> FP32: unpacks using hardware instructions
auto f32_unpacked = opus::cast<fp32_t>(fp8x4_val);

// FP32 -> FP4 (GFX950 only)
auto f4_packed = opus::cast<fp4_t>(fp32x8_val, scale);
```

---

## 9. MFMA (Matrix Core) Operations

### Creating an MFMA Instance

```cpp
using namespace opus;

// Direct construction
auto mma = make_mfma<fp16_t, fp16_t, fp32_t>(32_I, 32_I, 8_I);
// = mfma_f32_32x32x8_f16: 32x32 output, K=8 per instruction

auto mma = make_mfma<fp16_t, fp16_t, fp32_t>(16_I, 16_I, 16_I);
// = mfma_f32_16x16x16_f16: 16x16 output, K=16 per instruction

// With A/B operand swap (transposes output layout)
auto mma = make_mfma<fp16_t, fp16_t, fp32_t>(32_I, 32_I, 8_I,
                     mfma_adaptor_swap_ab{});

// From seq
auto mma = make_mfma<fp16_t, fp16_t, fp32_t>(seq<32, 32, 8>{});
```

### Supported MFMA Instructions

| Type Alias | dtypes (A, B, C) | MxNxK | Architecture |
|-----------|-------------------|-------|--------------|
| `mfma_f32_32x32x8_f16` | fp16, fp16, fp32 | 32x32x8 | GFX942, GFX950 |
| `mfma_f32_16x16x16_f16` | fp16, fp16, fp32 | 16x16x16 | GFX942, GFX950 |
| `mfma_f32_32x32x8_bf16` | bf16, bf16, fp32 | 32x32x8 | GFX942, GFX950 |
| `mfma_f32_16x16x16_bf16` | bf16, bf16, fp32 | 16x16x16 | GFX942, GFX950 |
| `mfma_f32_32x32x16_fp8_fp8` | fp8, fp8, fp32 | 32x32x16 | GFX942, GFX950 |
| `mfma_f32_16x16x32_fp8_fp8` | fp8, fp8, fp32 | 16x16x32 | GFX942, GFX950 |
| `mfma_f32_32x32x16_bf8_bf8` | bf8, bf8, fp32 | 32x32x16 | GFX942, GFX950 |
| `mfma_f32_16x16x32_bf8_bf8` | bf8, bf8, fp32 | 16x16x32 | GFX942, GFX950 |
| `mfma_f32_32x32x16_f16` | fp16, fp16, fp32 | 32x32x16 | GFX950 only |
| `mfma_f32_16x16x32_f16` | fp16, fp16, fp32 | 16x16x32 | GFX950 only |
| `mfma_f32_32x32x16_bf16` | bf16, bf16, fp32 | 32x32x16 | GFX950 only |
| `mfma_f32_16x16x32_bf16` | bf16, bf16, fp32 | 16x16x32 | GFX950 only |

GFX942 also supports "step-K" variants that chain multiple MFMA instructions to achieve larger effective K (e.g., 32x32x16 via two 32x32x8 calls).

### Executing MFMA

```cpp
// With accumulator (C = A * B + C)
auto c_out = mma(a_reg, b_reg, c_reg);

// Without accumulator (C = A * B)
auto c_out = mma(a_reg, b_reg);

// With cbsz/abid/blgp broadcast control
auto c_out = mma(a_reg, b_reg, c_reg, 0_I, 0_I, 0_I);
```

### Per-Thread Element Counts

For a 64-lane wavefront, each lane holds:

| | Formula | 32x32x8 (fp16) | 16x16x16 (fp16) |
|---|---------|----------------|-----------------|
| `elem_a` | M*K/64 | 4 | 4 |
| `elem_b` | N*K/64 | 4 | 4 |
| `elem_c` | M*N/64 | 16 | 4 |

---

## 10. Tiled MMA (Multi-Wave GEMM)

For block-level GEMM that spans multiple waves, use `make_tiled_mma`:

```cpp
// 2 waves along M, 2 along N, each wave repeats 2x along M
// Using 16x16x16 MFMA:
// Block tile = (2*2*16) x (1*2*16) x (1*1*16) = 64x32x16
auto tmma = make_tiled_mma<fp16_t, fp16_t, fp32_t>(
    seq<2, 1, 1>{},      // EXPAND: per-wave repeat M=2, N=1, K=1
    seq<2, 2, 1>{},       // TILE: wave grid M=2, N=2, K=1
    seq<16, 16, 16>{},    // WAVE: MFMA shape M=16, N=16, K=16
    mfma_adaptor_swap_ab{}
);
```

### Parameters

| Parameter | Meaning |
|-----------|---------|
| `EXPAND (seq<EM, EN, EK>)` | Per-wave repetition count in M/N/K |
| `TILE (seq<TM, TN, TK>)` | Number of waves in M/N/K grid |
| `WAVE (seq<WM, WN, WK>)` | MFMA instruction shape |

Block tile size = `(EM * TM * WM) x (EN * TN * WN) x (EK * TK * WK)`

### Execution

```cpp
// array-based (OPUS_TILE_CONTAINER == 1)
array<fp16x4_t, 2> a_tiles;
array<fp16x4_t, 1> b_tiles;
array<fp32x4_t, 2> c_tiles;
c_tiles = tmma(a_tiles, b_tiles, c_tiles);

// vector-based (OPUS_TILE_CONTAINER == 0)
fp16x8_t a_vec;    // All A data flattened
fp16x4_t b_vec;
fp32x8_t c_vec;
c_vec = tmma(a_vec, b_vec, c_vec);
```

---

## 11. Distributed Tensor Views (x/p/y dims)

GPU tensors are distributed across threads. Opus uses the `p_dim` / `y_dim` / `x_dim` terminology (adapted from ck_tile) to describe this distribution:

- **x-dim**: The original tensor dimensions (full tile view)
- **p-dim**: Dimensions partitioned across threads (parallel dimension)
- **y-dim**: Dimensions local to each thread (register dimension)

### Example: 48x32 Tensor across 64 Threads

```
         x[0]       x[1]
          v          v
tensor : [48      , 32]
view   : [[3,  16], [4,   8]]
           ^   ^     ^    ^
         y[0] p[0]  p[1] y[1]
```

- 8 contiguous elements per thread per row (y[1]=8)
- 4 threads cover one row (p[1]=4)
- 16 threads cover 16 rows (p[0]=16)
- Each thread repeats 3 times to cover 48 rows (y[0]=3)

### Adaptors

An adaptor struct declares shape and p/y dimension annotations:

```cpp
struct my_tile_adaptor {
    OPUS_H_D constexpr auto shape() {
        return opus::make_tuple(3_I, 16_I, 4_I, 8_I);
    }
    OPUS_H_D constexpr auto dim() {
        using namespace opus;
        return tuple<tuple<y_dim, p_dim>, tuple<p_dim, y_dim>>{};
    }
};
```

### Building Partition Layouts

```cpp
auto lane_id = threadIdx.x % 64;
auto x_stride = make_tuple(row_stride, 1_I);
auto p_coord = make_tuple(lane_id / 4_I, lane_id % 4_I);

auto u = make_layout(
    adaptor.shape(),
    unfold_x_stride(adaptor.dim(), adaptor.shape(), x_stride),
    unfold_p_coord(adaptor.dim(), p_coord)
);

// Now u(y0, y1) gives the global offset for this thread
auto offset = u(1, 0);    // y[0]=1, y[1]=0
```

### MFMA Adaptor Layout

MFMA adaptors automatically provide shape/dim for A, B, C operands:

```cpp
auto mma = make_mfma<fp16_t, fp16_t, fp32_t>(16_I, 16_I, 16_I);

// Get partition layouts with strides and thread coordinate
auto u_a = mma.layout_a(make_tuple(stride_k, 1_I), make_tuple(lane_id / 16_I, lane_id % 16_I));
auto u_b = mma.layout_b(make_tuple(stride_k, 1_I), make_tuple(lane_id / 16_I, lane_id % 16_I));
auto u_c = mma.layout_c(make_tuple(stride_n, 1_I), make_tuple(lane_id / 16_I, lane_id % 16_I));

// Packed layout (strides auto-computed)
auto u_a_packed = mma.layout_a_packed(p_coord);

// Query p-dim and y-dim shapes
auto p_a = mma.p_shape_a();    // Parallel dimensions
auto y_a = mma.y_shape_a();    // Per-thread dimensions
```

---

## 12. Utility Functions

### Static Loops

```cpp
// Compile-time unrolled loop
static_for<4>([&](auto i) {
    // i is number<0>, number<1>, number<2>, number<3>
    arr[i.value] = compute(i);
});

// Multi-dimensional static loop
static_ford<4, 8>([&](auto i, auto j) {
    // i in [0,4), j in [0,8)
});

// Loop with range
static_for([&](auto i) { ... }, 2, 8);      // i in [2, 8)
static_for([&](auto i) { ... }, 0, 10, 2);  // i in [0, 10) step 2
```

### DPP Operations

```cpp
// Data Parallel Primitives (intra-wavefront shuffle)
auto result = opus::mov_dpp(val, number<dpp_ctrl>{});
auto result = opus::upd_dpp(old_val, new_val, number<dpp_ctrl>{});
```

### Math

```cpp
opus::max(a, b)          // Uses __builtin_fmaxf for float
opus::min(a, b)          // Uses __builtin_fminf for float
opus::med3(a, b, c)      // Median of 3, uses __builtin_amdgcn_fmed3f
```

### Waitcnt

```cpp
opus::s_waitcnt(0_I, 0_I);           // Wait for all vmcnt and lgkmcnt
opus::s_waitcnt_vmcnt(0_I);          // Wait for global memory loads
opus::s_waitcnt_lgkmcnt(0_I);        // Wait for LDS operations
```

### Warp Size

```cpp
constexpr auto ws = opus::get_warp_size();  // 64 for GFX9
```

### Convenience Macros

```cpp
// Inside kernel function body
OPUS_USING_COMMON_TYPES
// Expands to:
//   using opus::operator""_I;
//   using p_dim = opus::p_dim;
//   using y_dim = opus::y_dim;

// At global scope (outside kernel)
OPUS_USING_COMMON_TYPES_ALL
// Additionally adds:
//   template<index_t I>     using num = opus::number<I>;
//   template<typename... T> using tup = opus::tuple<T...>;
//   template<index_t... Is> using seq = opus::seq<Is...>;
```

---

## 13. Usage in AITER Kernels

Opus is used in several AITER kernel implementations:

### RMSNorm + Quantization (`rmsnorm_quant_kernels.cu`)

```cpp
#include "aiter_opus_plus.h"  // AITER wrapper utilities

// Create typed vectors
using vec_i = opus::vector_t<DTYPE_I, thread_data_size>;
using vec_f = opus::vector_t<float, thread_data_size>;

// Gmem with OOB protection
auto buffer_i = opus::make_gmem<DTYPE_I>(input_ptr, oob * sizeof(DTYPE_I));
auto weight_buffer = opus::make_gmem<DTYPE_I>(weight, oob * sizeof(DTYPE_I));

// Vectorized load (via aiter_opus_plus helper)
vec_i data = load_vector_nbytes<DTYPE_I, thread_data_size,
                                load_chunk_bytes, load_aux, interleave>(
    buffer_i, row_offset);
```

### Top-K Selection (`topk_plain_kernels.cu`)

```cpp
#include "opus/opus.hpp"

// Use opus gmem for buffer loads with OOB safety
auto g = opus::make_gmem(ptr, size);
auto vals = g.load<vec_size>(offset);

// DPP-based bitonic sort within wavefront
auto swapped = opus::mov_dpp(val, number<dpp_ctrl>{});
```

### AITER Opus Plus Helpers (`aiter_opus_plus.h`)

AITER provides additional wrapper functions in `aiter_opus_plus.h` that build on opus:

- `load_vector_nbytes<T, vec_size, chunk_bytes, aux, interleave>()` — Chunked vectorized load with configurable chunk size and interleave pattern
- `store_vector_nbytes<T, DTYPE_I, vec_size, chunk_bytes, aux, interleave>()` — Chunked vectorized store with optional type conversion

These helpers handle cases where the total vector size exceeds the maximum single-instruction load width (16 bytes) by splitting into multiple smaller loads.

---

## Decision Tree: When to Use Opus

```
Need a GPU kernel for AMD?
├── Need pre-built kernel (GEMM, attention, etc.)?
│   └── Use AITER kernels or ck_tile directly
├── Need full layout transformation algebra?
│   └── Use ck_tile or cutlass/cute
├── Need vectorized buffer load/store?
│   └── Use opus::gmem / opus::smem ✓
├── Need MFMA with minimal boilerplate?
│   └── Use opus::make_mfma / opus::make_tiled_mma ✓
├── Need data type conversions (FP8, FP4, BF16)?
│   └── Use opus::cast<D>(src) ✓
├── Need simple index calculations for tiled access?
│   └── Use opus::make_layout ✓
└── Need distributed tensor view across threads?
    └── Use opus adaptors with p_dim/y_dim ✓
```

---

## Source Files

| File | Description |
|------|-------------|
| `csrc/include/opus/opus.hpp` | Single-header library (~1813 lines) |
| `csrc/include/opus/README.md` | Original README with GEMM example |
| `csrc/include/aiter_opus_plus.h` | AITER-specific helper utilities (chunked load/store) |
| `csrc/kernels/topk_plain_kernels.cu` | Top-K kernel using opus gmem and DPP |
| `csrc/kernels/rmsnorm_quant_kernels.cu` | RMSNorm+Quant kernel using opus types and gmem |
| `csrc/kernels/quant_kernels.cu` | Quantization kernels using opus |
| `csrc/kernels/cache_kernels.cu` | KV-cache kernels using opus |
