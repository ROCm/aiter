<!-- markdownlint-disable MD001 MD041 -->
<div align="center" id="sglangtop">
<img src="logo.png" alt="logo" width="400" margin="10px"></img>

## opus: AI (o)(p)erator Micro(u) (s)td
*Crafting the micro standard templates for AI Operators on ROCm*
</div>

## About
**opus** is a light weight templated C++ DSL designed to accelerate writting HIP/C++ based kernels on AMD GPU. It is highly inspired by project like [ck/ck_tile](https://github.com/ROCm/composable_kernel), [cutlass/cute](https://github.com/NVIDIA/cutlass), but with much simplified design and better maintainability.

It is a single-header file, hence only included very basic abstractions. Trade off must be made to include new concept into **opus**. One example is there is no `tensor` concept in **opus**, which usually contains both data provider (a pointer, or array/tuple for register) and layout descriptor (to help index calculation) inside one class. **opus** seperate them into 2 different classes, and still allow user to manually calculate index. As a consequence, the positioning of **opus** is `above hand written HIP kernel, below optimized template library like ck/cutlass`.

If you are looking for:
- AMDGPU data type declare, convert, 
- buffer load/store vectorization auto dispatch (instead of manually write by yourself)
- different matrix core instruction, and don't want change much code while switch between different mfma.
- various utility device function
- (optionally) some simple and easy to understand layout abstraction to boost index calculation.

then **opus** is a good choice for you.

However, if you are looking for:

- optimized gemm/fa/reduce/... kernel to use directly
- optimized gemm/fa/reduce/... device pipeline to reuse
- some layout system can describe any tensor transformation

then **opus** is not a good one, you may looking for `ck` or `aiter` kernels.

## Design
**opus** source code can be devided into two part (within a single file). The first half is some device independent structures, containers, utility functions. The second half is arch related device function like buffer load/store, mfma, etc... let's use a simple gemm as example to show case how to use **opus**

### naive gemm using opus
#### 1. vectorization load/store
First you may wait to load data from global memory. This can be done as simple as poninter dereference:
```
int offset_a = (threadIdx.x / 32 * 4) + (threadIdx.x % 32 * stride_a);
fp16x4_t v_a = *reinterpret_cast<const fp16x4_t*>(reinterpret_cast<const fp16_t*>(ptr_a) + offset_a);
```
For this naive example, we load data based on the matrix core layout requirement of A matrix (check [this blog](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores/README.html) for more detail about matrix core).

This is OK, but if we want to control vectorization (e.g. sometimes we have to load data pixel by pixel due to layout) we have to rewrite the code, like use more `if constexpr` to conditionally do the loading. If we think about `ck`, `cutlss`, or `triton`, we don't need to change code very much, this is powerful place of DSL.

in **opus** we can achieve so by:
```
// create fp16 gmem and load with vector size load<*>
auto g_a = opus::make_gmem(reinterpret_cast<const opus::fp16_t*>(ptr_a));
auto v_a = g_a.load<4>((threadIdx.x / 32 * 4) + (threadIdx.x % 32 * stride_a));

// alternatively, directly create a fp16x4 gmem
auto g_a = opus::make_gmem(reinterpret_cast<const opus::fp16x4_t*>(ptr_a));
auto v_a = g_a.load(((threadIdx.x / 32 * 4) + (threadIdx.x % 32 * stride_a)) / 4_I);
```
Note we use `auto` to hint the return loading data without knowing the vectorization before hand. Indeed the `gmem` structure will automatically do the vectorization load for you by `load<*>`/`store<*>`. What's more, it may utilize the buffer load OOB feature of AMD GPU for you, if you provide a second argument to tell the size of this buffer. Check [AMD GPU ISA](https://gpuopen.com/machine-readable-isa/) and `make_gmem()` api within `opus.hpp`

#### 2. layout for index calculation
**opus** provide a simple tensor descriptor to help calculate the address, name it `layout`. It uses very simple stride and coordinate to calculate the linear offset of a ND tensor:
```
int offset = index[0] * stride[0] + index[1] * stride[1] + index[2] * stride[2] + ...
```
here index/stride can be either static or dynamic variable.

This would be the first real step when you feel hand written is tedious and want some DSL to accelerate the index calculation. `create descriptor first, then doing the math` instead of expand your index calculation for every place, this is exactly a DSL is good at.

in **opus** you can create a layout and calculate address by:
```
auto u = opus::make_layout(opus::make_tuple(128, 64));
...
int offset = u(4, 8); // will return 4 * 64 + 8 * 1
```
the first arguement for `make_layout` is a `tuple`, to describe the shape of this tensor. If no more arguement provided, then assume this is a `packed tensor`, will internally calculate the stride based on this input shape.

#### 3. x-dim/p-dim/y-dim, how to describe tensor in a distributed way across multiple threads.
*optional if you don't want to introduce too many concept*

for GPU, it is natually need to consider in a multi-threaded way when dealing with a tensor, aka, one threads only responsible for a small portion of the tensor, multiple threads will collaboratively be responible for the whole tensor.

Suppose we have a `48x32` tensor from global, a wave with `64` threads want to load from this tensor. suppose every threads vectorized load `8` contiguous pixel, every row with `32` pixel will have `4` threads to load. The remaning `64/4=16` threads will responsible for load `16` rows. In the end, every threads need to repeat `48/16=3` times to finish such loading. By borrowing the `p/y/x` terms from [ck_tile](https://github.com/ROCm/composable_kernel/tree/develop/include/ck_tile), we can describe the layout partition in following psudo code:
```
         x[0]       x[1]
          v          v
tensor : [48      , 32]
view   : [[3,  16], [4,   8]]
           ^   ^     ^    ^
         y[0] p[0]  p[1] y[1]
```
- x-dim: view it as a whole tensor
- y-dim: dims that is within a single threads (inside register)
- p-dim: dims that need cross threads collaboratoin.

But `layout` structure we designed is so simple that it does not know the `x/y/p` information, how can we achieve such complex tensor description? This is through
1. use `underscore` internally, a special empty structure to hint the dims that need use weathre p or y dim (if you don't quite understand it, just skip it. this is internally trick)
2. use `adaptor` concept to provide more information

above example can be expressed like this:
```
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
auto offset = u(1, 0); // => get ofset at y[0] = 1, y[1] = 0 for each thread
```
Indeed **opus** support load/store with `layout` as argument, in which case will help you do all the dirty calculation. As of above example, usually need explcitly for loop to load `3` times with dwordx4. But with this feature, it can be simplified as:
```
auto g = opus::make_gmem(reinterpret_cast<const some_tile_dtype*>(ptr));

// originally we need do below for loop
some_vec_type v[3];
for(auto i = 0; i < 3; i++)
    v[i] = g.load<8>(u(i, 0));

// or, feed the layout to load(), then it will do the for loop for you.
auto v = g.load<8>(u);
```

#### 4. warp gemm and tiled mma
use `make_mfma()` to create a warp gemm instance, and use `make_tiled_mma()` to create a block gemm(multi-warp) instance. If you check the **opus** source code of mfma related code, you can see they all return a `adaptor` structure.
```
auto mma = opus::make_mfma()
```


While calling `make_layout`, the 1st arguement `shape` is usually from x-dim point of view. the 2nd optional arguement is `stride`, but from x-dim point of view. the 3rd optional arguement is `coordinate`, but from p-dim point of view. And the `operator()` which return the linear offset, is indeed from y-dim point of view. Above tensor can be described by:
```
using namespace opus;

// make 32x32x8 f16 matrix core
auto mma = make_mfma<fp16_t, fp16_t, fp32_t>(32_I, 32_I, 8_I);

// make 32x32x8 f16 matrix core, while a/b swapped
auto mma = make_mfma<fp16_t, fp16_t, fp32_t>(32_I, 32_I, 8_I, mfma_adaptor_swap_ab{});

// make 2x2 warp gemm of 16x16x16 mfma, a/b swapped, each wave repeat along m direction 2 times
// hence block tile: 64x32x16
auto mma = make_tiled_mma<fp16_t, fp16_t, fp32_t>(seq<2, 1, 1>{}, seq<2, 2, 1>{}, seq<16, 16, 16>{}, mfma_adaptor_swap_ab{});

```

check [this repo](https://github.com/carlushuang/gcnasm/tree/master/matrix_core_opus) for mfma example using **opus**

## C++ key feature used
1. static(constexpr)/dynamic variable
2. constexpr return type
3. local scratch
4. class inheritance (mainly 2 places. tuple use multi-inheritance implementation. adaptors use inheritance to overwrite layout & function call.)
5. function template partial specializatoin
6. recursive template expand
7. C++17 fold expresion
