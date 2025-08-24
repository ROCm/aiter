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


## `aiter::mha_fwd` supported arguments configuration
Note: For optimal performance, the input configuration preferentially matches the supported parameters of the asm kernel type.

| data_type    | hdim_q  | hdim_v  | seqlen_q      | seqlen_k          | mode           | mask_type                | general constraints            | shape&stride constraints                                                                       | kernel type | mi308 | mi300/325 | mi350/355         |
|--------------|---------|---------|---------------|-------------------|----------------|--------------------------|--------------------------------|------------------------------------------------------------------------------------------------|-------------|-------|-----------|-------------------|
| bf16         | 128     | 128     | [384,)        | equal to seqlen_q | batch or group | no_mask or causal        | bias, dropout is not supported | the shape&stride of q, k and v must be the same, the layout of q, k, v, o must be bshd or bhsd | asm         | y     | y         | lse must be true  |
| fp16 or bf16 | [0,32]  | [0,32]  | unconstrained | unconstrained     | batch or group | no_mask or causal or swa | unconstrained                  | unconstrained                                                                                  | ck          | y     | y         | y                 |
| fp16 or bf16 | (0,64]  | (0,64]  | unconstrained | unconstrained     | batch or group | no_mask or causal or swa | unconstrained                  | unconstrained                                                                                  | ck          | y     | y         | y                 |
| fp16 or bf16 | (0,128] | (0,128] | unconstrained | unconstrained     | batch or group | no_mask or causal or swa | unconstrained                  | unconstrained                                                                                  | ck          | y     | y         | y                 |
| fp16 or bf16 | (0,192] | (0,128] | unconstrained | unconstrained     | batch or group | no_mask or causal or swa | unconstrained                  | unconstrained                                                                                  | ck          | y     | y         | y                 |
| fp16 or bf16 | (0,256] | (0,256] | unconstrained | unconstrained     | batch or group | no_mask or causal or swa | unconstrained                  | unconstrained                                                                                  | ck          | y     | y         | y                 |


## `aiter::mha_bwd` supported arguments configuration
Note: For optimal performance, the input configuration preferentially matches the supported parameters of the asm kernel type.

| data_type    | hdim_q       | hdim_v          | mode           | mask_type                | dq_accumulation          | general constraints                                     | shape&stride constraints                                                                                                                                                                                                               | kernel type(asm/ck) | mi308 | mi300/325 | mi350/355                        |
|--------------|--------------|-----------------|----------------|--------------------------|--------------------------|---------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------|-------|-----------|----------------------------------|
| fp16 or bf16 | (128,192]/x8 | equal to hdim_q | batch or group | no_mask or causal        | atomic_f32               | bias, dbisa, dropout and deterministic is not supported | dq_acc only support BHSD                                                                                                                                                                                                               | asm                 | y     | y         | n                                |
| fp16 or bf16 | (64,128]/x8  | equal to hdim_q | batch          | no_mask or causal        | atomic_f32 or atomic_f16 | bias, dbisa, dropout and deterministic is not supported | dq_acc only support BHSD when dq_accumulation is atomic_f32. The shape&stride of q and do must be the same and the shape&stride of k and v must be the same and seqlen_q must be equal to seqlen_k when dq_accumulation is atomic_f16. | asm                 | y     | y         | bf16;hd128;sq == sk;sq % 256==0  |
| fp16 or bf16 | (64,128]/x8  | equal to hdim_q | group          | no_mask or causal        | atomic_f32               | bias, dbisa, dropout and deterministic is not supported | dq_acc only support BHSD                                                                                                                                                                                                               | asm                 | y     | y         | bf16;hd128;sq == sk;sq % 256==0  |
| fp16 or bf16 | 64           | equal to hdim_q | batch          | no_mask or causal        | atomic_f32 or atomic_f16 | bias, dbisa, dropout and deterministic is not supported | dq_acc only support BHSD when dq_accumulation is atomic_f32. The shape&stride of q and do must be the same and the shape&stride of k and v must be the same and seqlen_q must be equal to seqlen_k when dq_accumulation is atomic_f16. | asm                 | y     | y         | n                                |
| fp16 or bf16 | 64           | equal to hdim_q | group          | no_mask or causal        | atomic_f32               | bias, dbisa, dropout and deterministic is not supported | dq_acc only support BHSD                                                                                                                                                                                                               | asm                 | y     | y         | n                                |
| fp16 or bf16 | [0,32]       | [0,32]          | batch or group | no_mask or causal or swa | atomic_f32 or atomic_f16 | unconstrained                                           | unconstrained                                                                                                                                                                                                                          | ck                  | y     | y         | y                                |
| fp16 or bf16 | (0,64]       | (0,64]          | batch or group | no_mask or causal or swa | atomic_f32 or atomic_f16 | unconstrained                                           | unconstrained                                                                                                                                                                                                                          | ck                  | y     | y         | y                                |
| fp16 or bf16 | (0,128]      | (0,128]         | batch or group | no_mask or causal or swa | atomic_f32 or atomic_f16 | unconstrained                                           | unconstrained                                                                                                                                                                                                                          | ck                  | y     | y         | y                                |
| fp16 or bf16 | (0,256]      | (0,256]         | batch or group | no_mask or causal or swa | atomic_f32 or atomic_f16 | unconstrained                                           | unconstrained                                                                                                                                                                                                                          | ck                  | y     | y         | y                                |
