<!-- markdownlint-disable MD001 MD041 -->
<div align="center" id="sglangtop">
<img src="logo.png" alt="logo" width="400" margin="10px"></img>

## opus: AI (o)(p)erator Micro(u) (s)td
*Crafting the micro standard templates for AI Operators on ROCm*
</div>

## About
**opus** is a lightweight, templated C++ DSL designed to accelerate the development of HIP/C++ kernels for AMD GPUs. Inspired by projects such as [ck/ck_tile](https://github.com/ROCm/composable_kernel) and [cutlass/cute](https://github.com/NVIDIA/cutlass), **opus** adopts a significantly simplified design while prioritizing maintainability.

Distributed as a single-header library (`opus.hpp`), **opus** provides only essential abstractions. This constraint requires careful trade-offs when introducing new concepts. For instance, **opus** deliberately avoids a unified `tensor` class—which typically combines data providers (pointers or register arrays/tuples) with layout descriptors (for index calculation)—and instead separates them into two distinct classes. This design preserves the flexibility of manual index computation while maintaining clarity. As a result, **opus** positions itself **above hand-written HIP kernels** yet **below highly optimized template libraries like ck/cutlass**.

If you are looking for:
- AMDGPU data type declaration and conversion
- Automated vectorized buffer load/store dispatch (without manual implementation)
- Support for various matrix core instructions with minimal code changes when switching MFMA types
- A collection of utility device functions
- (Optional) Simple and intuitive layout abstractions to streamline index calculations

then **opus** is a good choice for you.

However, if you are looking for:

- Pre-optimized kernels (e.g., GEMM, attention, reduction) for direct use
- Reusable device-side pipelines for GEMM/attention/reduction
- A comprehensive layout system capable of describing arbitrary tensor transformations

then **opus** is not the right fit — you may be looking for alternatives like `ck` or `aiter` kernels.

## File structure

```
csrc/include/opus/
├── opus.hpp       # Single-header library (all you need to include)
├── logo.png       # Logo
└── README.md      # This file
```

## Usage

Include the header in your HIP/C++ source:
```cpp
#include "opus/opus.hpp"
```

No separate build step is required — just make sure `csrc/include/` is on your include path.

## Design
The **opus** source code is structured into two logical sections within a single header file:
- The first half contains device-independent structures, containers, and utility functions (number, seq, array, tuple, layout, etc.)
- The second half includes architecture-specific device functions, such as buffer load/store operations and MFMA instructions

Below, we illustrate the usage of **opus** through a naive GEMM example.

### Naive GEMM using opus

#### 1. Vectorized load/store
Loading data from global memory can be as simple as pointer dereferencing:
```cpp
int offset_a = (threadIdx.x / 32 * 4) + (threadIdx.x % 32 * stride_a);
fp16x4_t v_a = *reinterpret_cast<const fp16x4_t*>(reinterpret_cast<const fp16_t*>(ptr_a) + offset_a);
```
*For this example, we load data based on the matrix core layout of the A matrix (check [this blog](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores/README.html) for more detail about matrix cores).*

However, manually controlling vectorization across different layouts can lead to repetitive and error-prone code. With **opus**, the same operation becomes more expressive and adaptable:
```cpp
// Create fp16 gmem and load with vectorized load<*>
auto g_a = opus::make_gmem(reinterpret_cast<const opus::fp16_t*>(ptr_a));
auto v_a = g_a.load<4>((threadIdx.x / 32 * 4) + (threadIdx.x % 32 * stride_a));

// Alternatively, directly create a fp16x4 gmem
auto g_a = opus::make_gmem(reinterpret_cast<const opus::fp16x4_t*>(ptr_a));
auto v_a = g_a.load(((threadIdx.x / 32 * 4) + (threadIdx.x % 32 * stride_a)) / 4_I);
```
Note we use `auto` to hint the return loading data without knowing the vectorization beforehand. The `gmem` abstraction automatically handles vectorized load/store operations. Optionally, it can leverage AMD GPU's out-of-bounds (OOB) load features when a buffer size is provided as the 2nd argument for `make_gmem()`. Refer to the [AMD GPU ISA](https://gpuopen.com/machine-readable-isa/) and the `make_gmem()` API in `opus.hpp` for details.

#### 2. Layout for index calculation
**opus** provides a lightweight `layout` descriptor to simplify ND tensor address calculation. It computes linear offsets as:
```
offset = index[0] * stride[0] + index[1] * stride[1] + index[2] * stride[2] + ...
```
Here, indices and strides can be static or dynamic values. Using layouts helps abstract repetitive index calculations into reusable descriptors.
```cpp
auto u = opus::make_layout(opus::make_tuple(128, 64));
...
int offset = u(4, 8); // returns 4 * 64 + 8 * 1
```
If no strides are provided, `make_layout` assumes a packed (row-major) tensor and computes strides automatically based on the input shape.

#### 3. x-dim / p-dim / y-dim — distributed tensor views across threads
*(optional — skip if you don't want to introduce additional concepts)*

In GPU programming, tensors are often distributed across multiple threads. Consider loading a `48x32` tensor using a 64-thread wavefront:
- Each thread loads 8 contiguous elements per row
- 4 threads cover one row (32 elements)
- The remaining 16 threads load 16 rows
- Each thread repeats 3 times to cover all 48 rows

We adapt the `p/y/x` terminology from [ck_tile](https://github.com/ROCm/composable_kernel/tree/develop/include/ck_tile) to describe this distribution:
```
         x[0]       x[1]
          v          v
tensor : [48      , 32]
view   : [[3,  16], [4,   8]]
           ^   ^     ^    ^
         y[0] p[0]  p[1] y[1]
```
- **x-dim**: The original tensor dimensions
- **y-dim**: Dimensions handled within a single thread (register-level)
- **p-dim**: Dimensions requiring collaboration across threads

While the basic `layout` structure does not inherently understand `x/y/p` partitioning, **opus** enables such descriptions through:

1. Internal use of `underscore` placeholders to hint at p/y dimensions
2. The `adaptor` concept to provide additional structural information

The above example can be expressed like this:
```cpp
struct some_tile_adaptor{
    OPUS_H_D constexpr auto shape()  { return opus::make_tuple(3_I, 16_I, 4_I, 8_I); }
    OPUS_H_D constexpr auto dim()    { using namespace opus;
                                       return tuple<tuple<y_dim, p_dim>, tuple<p_dim, y_dim>>{};}
};

template<typename S, typename C>
OPUS_H_D constexpr auto partition_layout(some_tile_adaptor && a, S&& x_stride, C&& p_coord) {
    return opus::make_layout(a.shape(),
                             opus::unfold_x_stride(a.dim(), a.shape(), x_stride),
                             opus::unfold_p_coord(a.dim(), p_coord));
}

...
auto lane_id = threadIdx.x % 64;
auto s = opus::make_tuple(some_row_stride, 1_I);
auto c = opus::make_tuple(lane_id / 4_I, lane_id % 4_I);

auto u = partition_layout(some_tile_adaptor{}, s, c);
...
auto offset = u(1, 0); // get offset at y[0]=1, y[1]=0 for each thread
```
**opus** also supports direct load/store operations using `layout` objects, which automate the indexing logic. For instance, instead of manually looping over repetitions:
```cpp
auto g = opus::make_gmem(reinterpret_cast<const some_tile_dtype*>(ptr));

some_vec_type v[3];
for(auto i = 0; i < 3; i++)
    v[i] = g.load<8>(u(i, 0));
```
You can simply write:
```cpp
auto g = opus::make_gmem(reinterpret_cast<const some_tile_dtype*>(ptr));
auto v = g.load<8>(u);
```

#### 4. Warp GEMM and tiled MMA
Use `make_mfma()` to create a warp-level GEMM instance, and `make_tiled_mma()` for multi-warp (block-level) GEMM operations. These functions return `adaptor` structures that integrate seamlessly with the layout system.
1. The 1st argument `shape` is usually from the x-dim point of view.
2. The 2nd optional argument is `stride`, from x-dim point of view.
3. The 3rd optional argument is `coordinate`, from p-dim point of view.
4. Use `operator()` to issue the underlying matrix core instruction.

```cpp
using namespace opus;

// 32x32x8 f16 matrix core
auto mma = make_mfma<fp16_t, fp16_t, fp32_t>(32_I, 32_I, 8_I);

// 32x32x8 f16 matrix core, with A/B swapped
auto mma = make_mfma<fp16_t, fp16_t, fp32_t>(32_I, 32_I, 8_I, mfma_adaptor_swap_ab{});

// 2x2 warp GEMM of 16x16x16 MFMA, A/B swapped, each wave repeats 2x along M
// Block tile: 64x32x16
auto mma = make_tiled_mma<fp16_t, fp16_t, fp32_t>(seq<2, 1, 1>{}, seq<2, 2, 1>{}, seq<16, 16, 16>{}, mfma_adaptor_swap_ab{});

...
v_c = mma(v_a, v_b, v_c);
```

Check [this repo](https://github.com/carlushuang/gcnasm/tree/master/matrix_core_opus) for a complete MFMA example using **opus**.

## Tests and examples

Unit tests and working examples live under `op_tests/opus/`. These serve both as a test suite and as reference code for using **opus** APIs:

| Path | Description |
|------|-------------|
| `op_tests/opus/test_opus_basic.cpp` | Host-only C++ tests for core components: number, seq, array, tuple (including `merge_peepholed_tuple`), layout, static_for, type traits |
| `op_tests/opus/device/test_mfma.cu` | GPU kernel: 32x32x8 fp16 MFMA using `make_mfma`, `partition_layout_a/b/c`, `make_gmem` (gfx942 only) |
| `op_tests/opus/device/test_vector_add.cu` | GPU kernel: vectorized element-wise addition using `make_gmem` with templated `VECTOR_SIZE` |

To run all tests (host + device) inside a ROCm Docker container:
```bash
cd op_tests/opus
./run_tests_in_docker.sh
```

See [`op_tests/opus/README.md`](../../op_tests/opus/README.md) for full details on the test structure and how to add new device tests.

## C++ key features used
1. Static (`constexpr`) / dynamic variables
2. `constexpr` return types
3. Local scratch
4. Class inheritance (mainly in 2 places: tuple uses multi-inheritance; adaptors use inheritance to overwrite layout & function calls)
5. Function template partial specialization
6. Recursive template expansion
7. C++17 fold expressions
