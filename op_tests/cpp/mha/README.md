# aiter mha kernel

this is an example how to benchmark aiter mha fwd/bwd kernel through c++ API: `aiter::mha_fwd`, `aiter::mha_fwd_splitkv`, `aiter::mha_bwd`.

## build and run
We provide a simple script `build_mha.sh` to build the device library as well as a simple executable:
```
# this will build fwd_v3(asm) only
bash build_mha.sh fwd_v3

# this will build bwd_v3(asm) only
bash build_mha.sh bwd_v3

# this will build full fwd(asm + ck)
bash build_mha.sh fwd

# this will build full bwd(asm + ck)
bash build_mha.sh bwd

# this will build full fwd+bwd
bash build_mha.sh
```
Device library `libmha_fwd.so` and `libmha_bwd.so` will be built under current folder, and corresponding executables `benchmark_mha_fwd` and/or `benchmark_mha_bwd` will also be built. You can type `./benchmark_mha_fwd -?` to list all the supported arguments. You can also refer to the `smoke_test_*` script under this folder for a list of quick test.

To benchmark asm kernel, try following commands:
```
# Set this env before you run
export AITER_ASM_DIR={path_to_aiter}/hsa/{arch_name}/

# fwd_v3
./benchmark_mha_fwd -prec=bf16 -b=1 -h=64 -d=128 -s=8192 -iperm=1 -operm=1 -mask=1 -lse=1 -fwd_v3=1 -mode=0 -kname=1 -v=0

# bwd_v3 with atomic fp16
./benchmark_mha_bwd -prec=bf16 -b=1 -h=64 -d=128 -s=8192 -iperm=1 -operm=1 -mask=1 -bwd_v3=1 -v3_atomic_fp32=0 -v3_bf16_cvt=2 -mode=0 -kname=1 -v=0

# bwd_v3 with atomic fp32
./benchmark_mha_bwd -prec=bf16 -b=1 -h=64 -d=128 -s=8192 -iperm=1 -operm=1 -mask=1 -bwd_v3=1 -v3_atomic_fp32=1 -v3_bf16_cvt=2 -mode=0 -kname=1 -v=0
```

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

you can also call the corresponding executables `benchmark_mha_fwd` to check whether the arguments are supported by asm kernel with `-is_v3_check=1` condition, try following commands:
```
    ./benchmark_mha_fwd -prec=fp16 -b=1 -h=64 -d=128 -s=8192 -iperm=1 -operm=1 -mask=1 -lse=1 -fwd_v3=1 -mode=0 -kname=1 -v=0 -is_v3_check=1
```
| data_type    | hdim_q  | hdim_v  | mode           | mask_type                            | general constraints            | kernel type | mi308 | mi300/325 | mi350/355  |
|--------------|---------|---------|----------------|--------------------------------------|--------------------------------|-------------|-------|-----------|------------|
| bf16         | 128     | 128     | batch or group | no_mask or causal(mask_bottom_right) | bias, dropout is not supported | asm         | y     | y         | y          |
| bf16         | 192     | 128     | batch or group | no_mask or causal(mask_bottom_right) | bias, dropout is not supported | asm         | n     | n         | y          |
| fp16 or bf16 | [0,32]  | [0,32]  | batch or group | no_mask or causal or swa             | unconstrained                  | ck          | y     | y         | y          |
| fp16 or bf16 | (0,64]  | (0,64]  | batch or group | no_mask or causal or swa             | unconstrained                  | ck          | y     | y         | y          |
| fp16 or bf16 | (0,128] | (0,128] | batch or group | no_mask or causal or swa             | unconstrained                  | ck          | y     | y         | y          |
| fp16 or bf16 | (0,192] | (0,128] | batch or group | no_mask or causal or swa             | unconstrained                  | ck          | y     | y         | y          |
| fp16 or bf16 | (0,256] | (0,256] | batch or group | no_mask or causal or swa             | unconstrained                  | ck          | y     | y         | y          |


## `aiter::mha_bwd` supported arguments configuration
Note: For optimal performance, the input configuration preferentially matches the supported parameters of the asm kernel type.

you can also call the corresponding executables `benchmark_mha_bwd` to check whether the arguments are supported by asm kernel with `-is_v3_check=1` condition, try following commands:
```
    ./benchmark_mha_bwd -prec=bf16 -b=1 -h=64 -d=256 -s=8192 -iperm=1 -operm=1 -mask=1 -bwd_v3=1 -v3_atomic_fp32=0 -v3_bf16_cvt=2 -mode=0 -kname=1 -v=0 -is_v3_check=1
```

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


## the asm kernel performance of the attention forwards and attention backwards.
the performance data was tested under the conditions of BF16 and BSHD in batch mode.

| batch | q_nheads | kv_nheads | seqlen_q | seqlen_kv | hdim | casual | FWD(TFLOPS) |         |          | BWD(TFLOPS) |         |          |
|-------|----------|-----------|----------|-----------|------|--------|-------------|---------|----------|-------------|---------|----------|
|       |          |           |          |           |      |        | MI308       | MI300X  | MI355    | MI308       | MI300   | MI355    |
| 1     | 32       | 8         | 1024     | 1024      | 128  | 0      | 92.94       | 338.07  | 613.48   | 89.79       | 313.67  | 519.42   |
| 1     | 32       | 8         | 2048     | 2048      | 128  | 0      | 98.19       | 513.45  | 1194.46  | 116.81      | 269.19  | 701.34   |
| 1     | 32       | 8         | 4096     | 4096      | 128  | 0      | 114.77      | 527.73  | 1177.11  | 135.54      | 423.53  | 781.81   |
| 1     | 32       | 8         | 8192     | 8192      | 128  | 0      | 124.9       | 558.17  | 1396     | 139.25      | 481.28  | 818.43   |
| 1     | 32       | 8         | 10240    | 10240     | 128  | 0      | 127.09      | 549.73  | 1421.77  | 143.05      | 491.28  | 830.49   |
| 4     | 32       | 8         | 1024     | 1024      | 128  | 0      | 106.84      | 458.41  | 956.51   | 106.94      | 353.44  | 660.81   |
| 4     | 32       | 8         | 2048     | 2048      | 128  | 0      | 120.51      | 504.8   | 1092.82  | 128.7       | 430.81  | 745.42   |
| 4     | 32       | 8         | 4096     | 4096      | 128  | 0      | 123.21      | 577.16  | 1343.02  | 135.62      | 457.38  | 801.75   |
| 4     | 32       | 8         | 8192     | 8192      | 128  | 0      | 124.93      | 574.62  | 1407.46  | 144.99      | 458.72  | 830.84   |
| 4     | 32       | 8         | 10240    | 10240     | 128  | 0      | 127.21      | 584.66  | 1414.26  | 144.89      | 476.64  | 800.43   |
| 8     | 32       | 8         | 1024     | 1024      | 128  | 0      | 115.06      | 459.43  | 891.28   | 106.71      | 329.69  | 664.81   |
| 8     | 32       | 8         | 2048     | 2048      | 128  | 0      | 120.38      | 543.77  | 1175.5   | 128.94      | 426.56  | 757.61   |
| 8     | 32       | 8         | 4096     | 4096      | 128  | 0      | 123.43      | 567.82  | 1351.12  | 137.52      | 460.44  | 807.57   |
| 8     | 32       | 8         | 8192     | 8192      | 128  | 0      | 126.2       | 585.29  | 1406.47  | 145.26      | 475.56  | 834.32   |
| 8     | 32       | 8         | 10240    | 10240     | 128  | 0      | 127.27      | 577.5   | 1366.47  | 145.65      | 480.87  | 840.56   |
| 1     | 64       | 8         | 1024     | 1024      | 128  | 0      | 93.42       | 418.36  | 1003.73  | 107.01      | 266.06  | 644.69   |
| 1     | 64       | 8         | 2048     | 2048      | 128  | 0      | 111.9       | 485.45  | 1018.07  | 127.87      | 393.6   | 724.91   |
| 1     | 64       | 8         | 4096     | 4096      | 128  | 0      | 123.31      | 546.34  | 1305.83  | 135.02      | 470.15  | 788.39   |
| 1     | 64       | 8         | 8192     | 8192      | 128  | 0      | 124.8       | 591.37  | 1412.91  | 142.53      | 441.82  | 822.75   |
| 1     | 64       | 8         | 10240    | 10240     | 128  | 0      | 127.14      | 572.09  | 1417.43  | 87.79       | 460     | 831.34   |
| 1     | 64       | 8         | 16384    | 16384     | 128  | 0      | 125.88      | 547.78  | 1437.62  | 147.19      | 441.55  | 843.54   |
| 4     | 64       | 8         | 1024     | 1024      | 128  | 0      | 114.96      | 440.07  | 914.7    | 106.84      | 340.25  | 672.49   |
| 4     | 64       | 8         | 2048     | 2048      | 128  | 0      | 113.5       | 554.8   | 1201.6   | 128.82      | 425.74  | 757.48   |
| 4     | 64       | 8         | 4096     | 4096      | 128  | 0      | 123.38      | 573.6   | 1360.79  | 137.48      | 456.78  | 804.47   |
| 4     | 64       | 8         | 8192     | 8192      | 128  | 0      | 126.29      | 592.16  | 1407.58  | 142.24      | 468.71  | 798      |
| 4     | 64       | 8         | 10240    | 10240     | 128  | 0      | 127.32      | 578.93  | 1358.41  | 145.69      | 479.52  | 834.79   |
| 8     | 64       | 8         | 1024     | 1024      | 128  | 0      | 114.84      | 466.21  | 979.93   | 109.35      | 357.82  | 692.81   |
| 8     | 64       | 8         | 2048     | 2048      | 128  | 0      | 120.44      | 556.35  | 1250.96  | 130.75      | 430.07  | 764.92   |
| 8     | 64       | 8         | 4096     | 4096      | 128  | 0      | 124.53      | 578.99  | 1361.66  | 138.55      | 445.73  | 803.05   |
| 8     | 64       | 8         | 8192     | 8192      | 128  | 0      | 126.84      | 577.45  | 1322.77  | 145.25      | 475.07  | 806.58   |
| 8     | 64       | 8         | 10240    | 10240     | 128  | 0      | 127.28      | 571.39  | 1326.91  | 146.08      | 480.35  | 777.5    |
| 1     | 64       | 4         | 1024     | 1024      | 128  | 0      | 93.47       | 383.85  | 1017.04  | 106.81      | 264.63  | 637.29   |
| 1     | 64       | 4         | 2048     | 2048      | 128  | 0      | 111.94      | 506.89  | 1077.21  | 128.7       | 396.33  | 727.98   |
| 1     | 64       | 4         | 4096     | 4096      | 128  | 0      | 123.33      | 549.2   | 1299.05  | 135.01      | 467.24  | 787.19   |
| 1     | 64       | 4         | 8192     | 8192      | 128  | 0      | 124.84      | 591.77  | 1406.35  | 142.69      | 439.94  | 823.07   |
| 1     | 64       | 4         | 10240    | 10240     | 128  | 0      | 110.99      | 571.59  | 1429.39  | 112.46      | 459.64  | 834.05   |
| 1     | 64       | 4         | 16384    | 16384     | 128  | 0      | 125.83      | 549.09  | 1432.94  | 244.79      | 448.83  | 843.24   |
| 4     | 64       | 4         | 1024     | 1024      | 128  | 0      | 103.16      | 460.34  | 923.01   | 106.75      | 332.54  | 662.93   |
| 4     | 64       | 4         | 2048     | 2048      | 128  | 0      | 120.5       | 556.35  | 1224.58  | 128.7       | 424.12  | 757.93   |
| 4     | 64       | 4         | 4096     | 4096      | 128  | 0      | 123.37      | 575.69  | 1360.36  | 137.57      | 457.51  | 803.23   |
| 4     | 64       | 4         | 8192     | 8192      | 128  | 0      | 126.21      | 590.93  | 1411.19  | 111.34      | 469.72  | 816.86   |
| 4     | 64       | 4         | 10240    | 10240     | 128  | 0      | 127.19      | 582.64  | 1356.52  | 145.62      | 475.49  | 830.14   |
| 8     | 64       | 4         | 1024     | 1024      | 128  | 0      | 114.72      | 497.15  | 1016.32  | 109.51      | 360.39  | 694.07   |
| 8     | 64       | 4         | 2048     | 2048      | 128  | 0      | 120.53      | 556.22  | 1262.85  | 130.69      | 426.77  | 761.21   |
| 8     | 64       | 4         | 4096     | 4096      | 128  | 0      | 124.59      | 581.34  | 1362.68  | 138.47      | 438.77  | 796.47   |
| 8     | 64       | 4         | 8192     | 8192      | 128  | 0      | 126.85      | 583.23  | 1324     | 145.26      | 475.68  | 758.9    |
| 8     | 64       | 4         | 10240    | 10240     | 128  | 0      | 127.35      | 566.17  | 1325.23  | 145.75      | 478.88  | 841.68   |
| 1     | 32       | 8         | 1024     | 1024      | 128  | 1      | 64.25       | 130.62  | 233.12   | 57.58       | 166.78  | 210.315  |
| 1     | 32       | 8         | 2048     | 2048      | 128  | 1      | 81.105      | 255.105 | 577.28   | 86.205      | 295.865 | 479.925  |
| 1     | 32       | 8         | 4096     | 4096      | 128  | 1      | 90.94       | 467.805 | 949.325  | 115.68      | 296.025 | 713.075  |
| 1     | 32       | 8         | 8192     | 8192      | 128  | 1      | 110.24      | 522.68  | 1247.73  | 110.065     | 388.235 | 765.75   |
| 1     | 32       | 8         | 10240    | 10240     | 128  | 1      | 121.755     | 440.12  | 1200.645 | 139.91      | 244.705 | 759.32   |
| 4     | 32       | 8         | 1024     | 1024      | 128  | 1      | 66.925      | 334.005 | 720.995  | 87.115      | 226.39  | 465.04   |
| 4     | 32       | 8         | 2048     | 2048      | 128  | 1      | 92.77       | 419.435 | 809.835  | 101.415     | 330.23  | 431.525  |
| 4     | 32       | 8         | 4096     | 4096      | 128  | 1      | 111.245     | 486.73  | 1130.115 | 127.675     | 416.54  | 723.945  |
| 4     | 32       | 8         | 8192     | 8192      | 128  | 1      | 118.395     | 547.09  | 1318.92  | 134.495     | 422.835 | 775.46   |
| 4     | 32       | 8         | 10240    | 10240     | 128  | 1      | 121.725     | 527.705 | 1342.995 | 140.33      | 432.545 | 767.995  |
| 8     | 32       | 8         | 1024     | 1024      | 128  | 1      | 76.39       | 311.385 | 623.93   | 86.755      | 258.26  | 457.025  |
| 8     | 32       | 8         | 2048     | 2048      | 128  | 1      | 99.34       | 412.99  | 894.45   | 101.865     | 326.355 | 620.48   |
| 8     | 32       | 8         | 4096     | 4096      | 128  | 1      | 111.47      | 513.1   | 1166.875 | 127.69      | 409.05  | 726.06   |
| 8     | 32       | 8         | 8192     | 8192      | 128  | 1      | 118.39      | 537.36  | 1316.805 | 136.51      | 441.4   | 772.67   |
| 8     | 32       | 8         | 10240    | 10240     | 128  | 1      | 121.985     | 556.045 | 1334.865 | 142.125     | 443.78  | 794.245  |
| 1     | 64       | 8         | 1024     | 1024      | 128  | 1      | 66.72       | 228.54  | 432.565  | 75.24       | 242.43  | 370.42   |
| 1     | 64       | 8         | 2048     | 2048      | 128  | 1      | 81.265      | 392.425 | 936.435  | 100.865     | 257.855 | 598.985  |
| 1     | 64       | 8         | 4096     | 4096      | 128  | 1      | 103.68      | 474.385 | 1046.085 | 126.07      | 378.155 | 694.125  |
| 1     | 64       | 8         | 8192     | 8192      | 128  | 1      | 118.335     | 518.29  | 1300.105 | 126.91      | 433.285 | 765.21   |
| 1     | 64       | 8         | 10240    | 10240     | 128  | 1      | 120.985     | 510.895 | 1338.005 | 140.2       | 447.995 | 788.92   |
| 1     | 64       | 8         | 16384    | 16384     | 128  | 1      | 122.53      | 541.55  | 1392.485 | 143.965     | 412.04  | 808.93   |
| 4     | 64       | 8         | 1024     | 1024      | 128  | 1      | 76.5        | 326.51  | 638.705  | 86.64       | 266.9   | 470.95   |
| 4     | 64       | 8         | 2048     | 2048      | 128  | 1      | 99.485      | 425.735 | 899.845  | 100.995     | 326.805 | 621.295  |
| 4     | 64       | 8         | 4096     | 4096      | 128  | 1      | 111.395     | 513.79  | 1174.92  | 127.65      | 391.235 | 722.205  |
| 4     | 64       | 8         | 8192     | 8192      | 128  | 1      | 118.65      | 540.515 | 1319.225 | 136.38      | 434.645 | 774.25   |
| 4     | 64       | 8         | 10240    | 10240     | 128  | 1      | 121.875     | 557.475 | 1337.965 | 142.075     | 442.51  | 792.12   |
| 8     | 64       | 8         | 1024     | 1024      | 128  | 1      | 81.595      | 321.865 | 626.57   | 87.285      | 265.08  | 484.34   |
| 8     | 64       | 8         | 2048     | 2048      | 128  | 1      | 99.77       | 452.03  | 963.165  | 104.115     | 347.89  | 630.43   |
| 8     | 64       | 8         | 4096     | 4096      | 128  | 1      | 111.495     | 509.255 | 1190.295 | 129.69      | 402.18  | 710.905  |
| 8     | 64       | 8         | 8192     | 8192      | 128  | 1      | 119.465     | 550.67  | 1311.955 | 137.1       | 432.715 | 772.605  |
| 8     | 64       | 8         | 10240    | 10240     | 128  | 1      | 121.89      | 547.05  | 1313.695 | 143.015     | 439.785 | 792.91   |
| 1     | 64       | 4         | 1024     | 1024      | 128  | 1      | 66.68       | 229.09  | 421.615  | 75.25       | 238.755 | 376.975  |
| 1     | 64       | 4         | 2048     | 2048      | 128  | 1      | 81.055      | 407.525 | 949.635  | 100.855     | 254.375 | 580.43   |
| 1     | 64       | 4         | 4096     | 4096      | 128  | 1      | 103.725     | 476.26  | 1058.9   | 126.805     | 384.585 | 705.6    |
| 1     | 64       | 4         | 8192     | 8192      | 128  | 1      | 118.32      | 519.32  | 1318.15  | 134.155     | 442.955 | 768.16   |
| 1     | 64       | 4         | 10240    | 10240     | 128  | 1      | 121.605     | 515.275 | 1348.155 | 140.175     | 459.745 | 785.905  |
| 1     | 64       | 4         | 16384    | 16384     | 128  | 1      | 122.395     | 544.1   | 1398.14  | 143.72      | 339.695 | 809.685  |
| 4     | 64       | 4         | 1024     | 1024      | 128  | 1      | 76.595      | 314.82  | 661.045  | 87.235      | 264.795 | 470.865  |
| 4     | 64       | 4         | 2048     | 2048      | 128  | 1      | 99.635      | 426.77  | 896.095  | 101.77      | 331.95  | 620.01   |
| 4     | 64       | 4         | 4096     | 4096      | 128  | 1      | 111.395     | 524.585 | 1182.87  | 127.67      | 405.02  | 713.075  |
| 4     | 64       | 4         | 8192     | 8192      | 128  | 1      | 118.47      | 540.935 | 1324.275 | 136.485     | 430.95  | 749.805  |
| 4     | 64       | 4         | 10240    | 10240     | 128  | 1      | 121.77      | 560.63  | 1346.46  | 142.045     | 441.345 | 780.665  |
| 8     | 64       | 4         | 1024     | 1024      | 128  | 1      | 82.15       | 348.76  | 663.48   | 87.62       | 267.48  | 493.61   |
| 8     | 64       | 4         | 2048     | 2048      | 128  | 1      | 99.775      | 461.89  | 983.355  | 104.215     | 352.7   | 626.72   |
| 8     | 64       | 4         | 4096     | 4096      | 128  | 1      | 111.58      | 513.795 | 1196.675 | 129.68      | 402.68  | 701.24   |
| 8     | 64       | 4         | 8192     | 8192      | 128  | 1      | 119.695     | 552.78  | 1318.92  | 137.54      | 434.51  | 774.87   |
| 8     | 64       | 4         | 10240    | 10240     | 128  | 1      | 121.84      | 548.65  | 1313.945 | 143.005     | 435.745 | 793.095  |
