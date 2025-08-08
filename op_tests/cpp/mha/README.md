# aiter mha kernel

this is an example how to benchmark aiter mha fwd/bwd kernel through c++ API: `aiter::mha_fwd`, `aiter::mha_fwd_splitkv`, `aiter::mha_bwd`.

## build and run
We provide a simple script `build_mha.sh` to build the device library as well as a simple executable:
```
# this will build fwd+bwd
bash build_mha.sh

# this will build fwd only
bash build_mha.sh fwd

# this will build bwd only
bash build_mha.sh bwd
```
Device library `libmha_fwd.so` and `libmha_bwd.so` will be built under current folder, and corresponding executables `benchmark_mha_fwd` and/or `benchmark_mha_bwd` will also be built. You can type `./benchmark_mha_fwd -?` to list all the supported arguments. You can also refer to the `smoke_test_*` script under this folder for a list of quick test.

## how to build/link aiter mha in your c++ project
We recommend you download the source code of `aiter` and put it under the `3rdparty` submodule folder of your project (you don't need to install `aiter`). We use a way simliar to [cpp_extension](https://github.com/pytorch/pytorch/blob/main/torch/utils/cpp_extension.py) to build the device kernel library without `torch` dependency (you don't need to install `torch`), so it's easy to embed `aiter` into other project.

Basically the build process will be similiar to that inside `build_mha.sh` script.

First, you need to build the device kernel into a `so`, which is done by a python `compile.py` inside this folder.
```
python3 compile.py
```
you can also call this python script from different directory, the generated `.so` will always under current directory.

Second, link the `.so` into your executable and compile. You need specify the correct path through `-L` inorder to link to the device lib. You also need to specify the include directory through `-I`, for this example you need set `$TOP_DIR/csrc/include` for the `aiter` API header, and the dependent ck header `$TOP_DIR/3rdparty/composable_kernel/include` and `$TOP_DIR/3rdparty/composable_kernel/example/ck_tile/01_fmha/`. Please refer to `build_mha.sh` for detailed command

## `aiter::mha_bwd` supported arguments configuration
Note: For optimal performance, the input configuration preferentially matches the supported parameters of the asm kernel type.

| No. | hdim_q       | hdim_v          | q_dtype_str  | is_group_mode | mask_type   | bias_type   | has_dbias     | p_drop | deterministic | use_ext_asm   | is_v3_atomic_fp32 | how_v3_bf16_cvt                                   | shape&stride constraints                                                                                                                                                                                                        | kernel type(asm/ck) | mi308 | mi300/325 | mi350/355                        |
|-----|--------------|-----------------|--------------|---------------|-------------|-------------|---------------|--------|---------------|---------------|-------------------|---------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------|-------|-----------|----------------------------------|
| 1   | (128,192]/x8 | equal to hdim_q | fp16 or bf16 | FALSE         | 0 or 1 or 2 | 0           | FALSE         | 0.0    | FALSE         | TRUE          | TRUE              | 0 or 1 or 2(needed only when q_dtype_str is bf16) | dq_acc only support BHSD                                                                                                                                                                                                        | asm                 | y     | y         | n                                |
| 2   | (128,192]/x8 | equal to hdim_q | fp16 or bf16 | TRUE          | 0 or 1      | 0           | FALSE         | 0.0    | FALSE         | TRUE          | TRUE              | 0 or 1 or 2(needed only when q_dtype_str is bf16) | dq_acc only support BHSD                                                                                                                                                                                                        | asm                 | y     | y         | n                                |
| 3   | (64,128]/x8  | equal to hdim_q | fp16 or bf16 | FALSE         | 0 or 1 or 2 | 0           | FALSE         | 0.0    | FALSE         | TRUE          | TRUE or FALSE     | 0 or 1 or 2(needed only when q_dtype_str is bf16) | dq_acc only support BHSD when is_v3_atomic_fp32 is TRUE. The shape&stride of q and do must be the same and the shape&stride of k and v must be the same and seqlen_q must be equal to seqlen_k when is_v3_atomic_fp32 is FALSE. | asm                 | y     | y         | bf16;hd128;sq == sk;sq % 256==0  |
| 4   | (64,128]/x8  | equal to hdim_q | fp16 or bf16 | TRUE          | 0 or 1 or 2 | 0           | FALSE         | 0.0    | FALSE         | TRUE          | TRUE              | 0 or 1 or 2(needed only when q_dtype_str is bf16) | dq_acc only support BHSD                                                                                                                                                                                                        | asm                 | y     | y         | bf16;hd128;sq == sk;sq % 256==0  |
| 5   | 64           | equal to hdim_q | fp16 or bf16 | TRUE or FALSE | 0           | 0           | FALSE         | 0.0    | FALSE         | TRUE          | TRUE or FALSE     | 0 or 1 or 2(needed only when q_dtype_str is bf16) | dq_acc only support BHSD when is_v3_atomic_fp32 is TRUE. The shape&stride of q and do must be the same and the shape&stride of k and v must be the same and seqlen_q must be equal to seqlen_k when is_v3_atomic_fp32 is FALSE. | asm                 | y     | y         | n                                |
| 6   | 64           | equal to hdim_q | fp16 or bf16 | TRUE          | 1           | 0           | FALSE         | 0.0    | FALSE         | TRUE          | TRUE              | 0 or 1 or 2(needed only when q_dtype_str is bf16) | dq_acc only support BHSD                                                                                                                                                                                                        | asm                 | y     | y         | n                                |
| 7   | 64           | equal to hdim_q | fp16 or bf16 | FALSE         | 1 or 2      | 0           | FALSE         | 0.0    | FALSE         | TRUE          | TRUE              | 0 or 1 or 2(needed only when q_dtype_str is bf16) | dq_acc only support BHSD                                                                                                                                                                                                        | asm                 | y     | y         | n                                |
| 8   | 64           | equal to hdim_q | fp16 or bf16 | TRUE or FALSE | 1 or 2      | 0           | FALSE         | 0.0    | FALSE         | TRUE          | FALSE             | 0 or 1 or 2(needed only when q_dtype_str is bf16) | the shape&stride of q and do must be the same and the shape&stride of k and v must be the same and seqlen_q must be equal to seqlen_k                                                                                           | asm                 | y     | y         | n                                |
| 9   | [0,32]       | [0,32]          | fp16 or bf16 | TRUE or FALSE | 0 or 1 or 2 | 0 or 1 or 2 | TRUE or FALSE | [0,1)  | TRUE or FALSE | UNCONSTRAINED | UNCONSTRAINED     | UNCONSTRAINED                                     | UNCONSTRAINED                                                                                                                                                                                                                   | ck                  | y     | y         | y                                |
| 10  | (0,64]       | (0,64]          | fp16 or bf16 | TRUE or FALSE | 0 or 1 or 2 | 0 or 1 or 2 | TRUE or FALSE | [0,1)  | TRUE or FALSE | UNCONSTRAINED | UNCONSTRAINED     | UNCONSTRAINED                                     | UNCONSTRAINED                                                                                                                                                                                                                   | ck                  | y     | y         | y                                |
| 11  | (0,128]      | (0,128]         | fp16 or bf16 | TRUE or FALSE | 0 or 1 or 2 | 0 or 1 or 2 | TRUE or FALSE | [0,1)  | TRUE or FALSE | UNCONSTRAINED | UNCONSTRAINED     | UNCONSTRAINED                                     | UNCONSTRAINED                                                                                                                                                                                                                   | ck                  | y     | y         | y                                |
| 12  | (0,256]      | (0,256]         | fp16 or bf16 | TRUE or FALSE | 0 or 1 or 2 | 0 or 1 or 2 | TRUE or FALSE | [0,1)  | TRUE or FALSE | UNCONSTRAINED | UNCONSTRAINED     | UNCONSTRAINED                                     | UNCONSTRAINED                                                                                                                                                                                                                   | ck                  | y     | y         | y                                |


## `aiter::mha_fwd` supported arguments configuration
Note: For optimal performance, the input configuration preferentially matches the supported parameters of the asm kernel type.

| No. | hdim_q  | hdim_v  | q_dtype_str  | is_group_mode | mask_type   | bias_type   | has_lse       | p_drop | use_ext_asm   | seqlen_q      | seqlen_k          | kernel type | mi308 | mi300/325 | mi350/355  |
|-----|---------|---------|--------------|---------------|-------------|-------------|---------------|--------|---------------|---------------|-------------------|-------------|-------|-----------|------------|
| 1   | 128     | 128     | bf16         | TRUE or FALSE | 0 or 1 or 2 | 0           | TRUE          | 0.0    | TRUE          | [384,)        | equal to seqlen_q | asm         | y     | y         | y          |
| 2   | 128     | 128     | bf16         | TRUE or FALSE | 0 or 1 or 2 | 0           | FALSE         | 0.0    | TRUE          | [384,)        | equal to seqlen_q | asm         | y     | y         | n          |
| 3   | [0,32]  | [0,32]  | fp16 or bf16 | TRUE or FALSE | 0 or 1 or 2 | 0 or 1 or 2 | TRUE or FALSE | [0,1)  | UNCONSTRAINED | UNCONSTRAINED | UNCONSTRAINED     | ck          | y     | y         | y          |
| 4   | (0,64]  | (0,64]  | fp16 or bf16 | TRUE or FALSE | 0 or 1 or 2 | 0 or 1 or 2 | TRUE or FALSE | [0,1)  | UNCONSTRAINED | UNCONSTRAINED | UNCONSTRAINED     | ck          | y     | y         | y          |
| 5   | (0,128] | (0,128] | fp16 or bf16 | TRUE or FALSE | 0 or 1 or 2 | 0 or 1 or 2 | TRUE or FALSE | [0,1)  | UNCONSTRAINED | UNCONSTRAINED | UNCONSTRAINED     | ck          | y     | y         | y          |
| 6   | (0,192] | (0,128] | fp16 or bf16 | TRUE or FALSE | 0 or 1 or 2 | 0 or 1 or 2 | TRUE or FALSE | [0,1)  | UNCONSTRAINED | UNCONSTRAINED | UNCONSTRAINED     | ck          | y     | y         | y          |
| 7   | (0,256] | (0,256] | fp16 or bf16 | TRUE or FALSE | 0 or 1 or 2 | 0 or 1 or 2 | TRUE or FALSE | [0,1)  | UNCONSTRAINED | UNCONSTRAINED | UNCONSTRAINED     | ck          | y     | y         | y          |
