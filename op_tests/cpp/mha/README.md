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
export AITER_ASM_DIR={path_to_aiter}/hsa/

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

you can also call the executable `fwd.exe` to check whether the arguments are supported by the asm kernel with the `-is_v3_check=1` condition, try following commands:
```
    ./fwd.exe -prec=bf16 -b=1 -h=64 -d=128 -s=8192 -iperm=1 -operm=1 -mask=1 -lse=1 -fwd_v3=1 -mode=0 -kname=1 -v=0 -is_v3_check=1
```
`causal` below always means `window_size_left == -1 && window_size_right == 0`. The asm and opus kernels are compiled for `mask_bottom_right`; `mask_top_left` is only accepted when `seqlen_q == seqlen_k` (the two are equivalent there). `fp8bf16` means fp8 q/k/v with a bf16 output, and it requires the fp32 `q/k/v_descale` buffers to be set.

| data_type    | hdim_q  | hdim_v  | mode           | mask_type                            | general constraints                                | kernel type | mi308 | mi300/325 | mi350/355  |
|--------------|---------|---------|----------------|--------------------------------------|----------------------------------------------------|-------------|-------|-----------|------------|
| bf16         | 128     | 128     | batch or group | no_mask or causal(mask_bottom_right) | bias, dropout and swa are not supported            | asm         | y     | y         | y          |
| bf16         | 192     | 128     | batch or group | no_mask or causal(mask_bottom_right) | bias, dropout and swa are not supported            | asm         | y     | y         | y          |
| fp8bf16      | 128     | 128     | batch or group | no_mask or causal(mask_bottom_right) | same as above; descale of q/k/v is required        | asm         | y     | y         | y          |
| fp8bf16      | 256     | 256     | batch or group | no_mask or causal(mask_bottom_right) | same as above; descale of q/k/v is required        | asm         | n     | n         | y          |
| bf16         | 128     | 128     | batch          | no_mask or causal(mask_bottom_right) | bias, dropout and swa are not supported            | opus        | n     | n         | y          |
| bf16         | 192     | 128     | batch or group | no_mask or causal(mask_bottom_right) | bias, dropout and swa are not supported            | opus        | n     | n         | y          |
| fp16 or bf16 | [0,32]  | [0,32]  | batch or group | no_mask or causal or swa             | unconstrained                                      | ck          | y     | y         | y          |
| fp16 or bf16 | (0,64]  | (0,64]  | batch or group | no_mask or causal or swa             | unconstrained                                      | ck          | y     | y         | y          |
| fp16 or bf16 | (0,80]  | (0,96]  | batch or group | no_mask or causal or swa             | unconstrained                                      | ck          | y     | y         | y          |
| fp16 or bf16 | (0,96]  | (0,128] | batch or group | no_mask or causal or swa             | unconstrained                                      | ck          | y     | y         | y          |
| fp16 or bf16 | (0,128] | (0,128] | batch or group | no_mask or causal or swa             | unconstrained                                      | ck          | y     | y         | y          |
| fp16 or bf16 | (0,192] | (0,128] | batch or group | no_mask or causal or swa             | unconstrained                                      | ck          | y     | y         | y          |
| fp16 or bf16 | (0,192] | (0,192] | batch or group | no_mask or causal or swa             | unconstrained                                      | ck          | y     | y         | y          |
| fp16 or bf16 | (0,256] | (0,256] | batch or group | no_mask or causal or swa             | unconstrained                                      | ck          | y     | y         | y          |
| fp8bf16      | (0,128] | (0,128] | batch or group | no_mask or causal or swa             | descale of q/k/v is required                       | ck          | y     | y         | y          |
| fp8bf16      | (0,192] | (0,128] | batch or group | no_mask or causal or swa             | descale of q/k/v is required                       | ck          | y     | y         | y          |

Notes:
* The ck rows are matched top-down: the first row whose `hdim_q`/`hdim_v` both fit is the one that gets dispatched.
* `logits_soft_cap` and the attention sink are only implemented by the ck kernels; the asm and opus paths do not guard against them, so pass `fwd_v3=0` (or leave it at the default) when you need them.
* `-v3_bf16_cvt` (0:RTNE, 1:RTNA, 2:RTZ) only affects the gfx942 asm kernels. All three variants exist for `bf16`, while `fp8bf16` on gfx942 only ships the RTNA(=1) variant. gfx950 has a single variant and ignores this flag.
* The opus rows are **not** reachable through `aiter::mha_fwd`. They have their own entry point, `fmha_fwd_bf16_opus_fwd`, which `fwd.exe` calls with `-fwd_v3=2`. bias, dropout, `logits_soft_cap` and the attention sink are not parameters of that entry point at all, so the API cannot be handed them by mistake — but `fwd.exe` still accepts `-bias`, `-p_drop`, `-logits_soft_cap`, `-qscale` and a non-bf16 `-prec` under `-fwd_v3=2` and passes the buffers down unchanged, which makes the reported number describe something other than what was asked for. A head-dim pair outside the two rows above, group mode on the D=128 kernel, and an over-large kv extent are refused and print `not supported yet`.
* The opus kernels are compiled for gfx950 only: on any other arch the kernel template expands to an empty stub, and nothing checks the arch at runtime, so a call there returns without writing `out`.
* The D=128 opus kernel needs the kv byte extent (`seqlen_k * max(k, v seqlen-stride) * 2`) to stay below 2^32, because a larger one wraps the async-load offset. The 192/128 kernel rebases its buffer descriptors per tile and has no such limit.
* q/k/v/out must be contiguous along the head dim; the remaining strides are free, so both bshd and bhsd work. `-vlayout=c` does not (opus reads V row-major over the sequence).


## `aiter::mha_bwd` supported arguments configuration
Note: For optimal performance, the input configuration preferentially matches the supported parameters of the asm kernel type.

you can also call the executable `bwd.exe` to check whether the arguments are supported by the asm kernel with the `-v3_api_check=1` condition, try following commands:
```
    ./bwd.exe -prec=bf16 -b=1 -h=64 -d=128 -s=8192 -iperm=1 -operm=1 -mask=1 -bwd_v3=1 -v3_atomic_fp32=0 -v3_bf16_cvt=2 -mode=0 -kname=1 -v=0 -v3_api_check=1
```
Unlike fwd, the bwd asm kernels have separate `mask_top_left` and `mask_bottom_right` instances, so `causal` below covers both unless stated otherwise. The generic mask (`-mask=g:y,x`) is never supported by asm. `dq_acc` is no longer supplied by the caller: it is allocated internally through `mha_bwd_args::workspace_alloc`.

| data_type    | hdim_q       | hdim_v          | mode           | mask_type                | dq_accumulation          | general constraints                                                       | shape&stride constraints                                                                                                                                                                                          | kernel type(asm/ck) | mi308 | mi300/325 | mi350/355 |
|--------------|--------------|-----------------|----------------|--------------------------|--------------------------|---------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------|-------|-----------|-----------|
| fp16 or bf16 | (128,192]/x8 | equal to hdim_q | batch or group | no_mask or causal        | atomic_f32               | bias, dbias, dropout and deterministic is not supported                   | unconstrained                                                                                                                                                                                                     | asm                 | y     | y         | y         |
| fp16 or bf16 | (64,128]/x8  | equal to hdim_q | batch or group | no_mask or causal        | atomic_f32               | bias, dbias, dropout and deterministic is not supported                   | unconstrained                                                                                                                                                                                                     | asm                 | y     | y         | y         |
| fp16 or bf16 | (64,128]/x8  | equal to hdim_q | batch          | swa                      | atomic_f32               | bias, dbias, dropout and deterministic is not supported                   | unconstrained                                                                                                                                                                                                     | asm                 | y     | y         | n         |
| fp16 or bf16 | (64,128]/x8  | equal to hdim_q | batch          | no_mask or causal_top_left | atomic_f16             | bias, dbias, dropout and deterministic is not supported                   | seqlen_q == seqlen_k and seqlen_k % 64 == 0. The shape&stride of q and do must be the same, the shape&stride of k and v must be the same, and dk/dv must keep the nhead stride of k/v.                             | asm                 | y     | y         | n         |
| fp16 or bf16 | (64,128]/x8  | equal to hdim_q | batch or group | no_mask or causal        | atomic_f16               | bias, dbias, dropout and deterministic is not supported                   | unconstrained                                                                                                                                                                                                     | asm                 | n     | n         | y         |
| fp16 or bf16 | 192          | 128             | batch          | no_mask or causal        | atomic_f32 or atomic_f16 | bias, dbias, dropout and deterministic is not supported                   | unconstrained                                                                                                                                                                                                     | asm                 | n     | n         | y         |
| fp16 or bf16 | 64           | equal to hdim_q | batch or group | no_mask or causal        | atomic_f32               | bias, dbias, dropout and deterministic is not supported                   | unconstrained                                                                                                                                                                                                     | asm                 | y     | y         | y         |
| fp16 or bf16 | 64           | equal to hdim_q | batch          | no_mask or causal_top_left | atomic_f16             | bias, dbias, dropout and deterministic is not supported                   | seqlen_q == seqlen_k and seqlen_k % 64 == 0. The shape&stride of q and do must be the same, the shape&stride of k and v must be the same, and dk/dv must keep the nhead stride of k/v.                             | asm                 | y     | y         | y         |
| fp16 or bf16 | [0,32]       | [0,32]          | batch or group | no_mask or causal or swa | atomic_f32 or atomic_f16 | unconstrained                                                             | unconstrained                                                                                                                                                                                                     | ck                  | y     | y         | y         |
| fp16 or bf16 | (0,64]       | (0,64]          | batch or group | no_mask or causal or swa | atomic_f32 or atomic_f16 | unconstrained                                                             | unconstrained                                                                                                                                                                                                     | ck                  | y     | y         | y         |
| fp16 or bf16 | (0,96]       | (0,96]          | batch or group | no_mask or causal or swa | atomic_f32 or atomic_f16 | unconstrained                                                             | unconstrained                                                                                                                                                                                                     | ck                  | y     | y         | y         |
| fp16 or bf16 | (0,128]      | (0,128]         | batch or group | no_mask or causal or swa | atomic_f32 or atomic_f16 | unconstrained                                                             | unconstrained                                                                                                                                                                                                     | ck                  | y     | y         | y         |
| fp16 or bf16 | (0,256]      | (0,256]         | batch or group | no_mask or causal or swa | atomic_f32 or atomic_f16 | unconstrained                                                             | unconstrained                                                                                                                                                                                                     | ck                  | y     | y         | y         |

Notes:
* All asm rows additionally require `hdim_q % 8 == 0 && hdim_v % 8 == 0`. `hdim_q` is padded up to 64/128/192 internally, and the `hdim_q == 64` bucket has no padded-hdim instance, so a head dim below 64 always falls back to ck.
* The rows marked `causal_top_left` have no `mask_bottom_right` instance. On gfx942 a bottom-right causal request is remapped to top-left, which is legal because those rows already require `seqlen_q == seqlen_k`; on gfx950 (`hdim_q == 64`, `atomic_f16`) there is no such remap and the bottom-right case falls back to ck.
* `-v3_bf16_cvt` (0:RTNE, 1:RTNA, 2:RTZ) picks the float→bf16 rounding variant of the bf16 dqdkdv and dq_convert instances. Every gfx942 bf16 instance is rounding-specific; on gfx950 only the `hdim_q == hdim_v == 192` and `hdim_q == hdim_v == 64` dqdkdv instances are, and all the fp16 instances are rounding-agnostic.
* gfx1250 is also dispatched to asm, but only for `bf16`, `hdim_q == hdim_v == 128`, batch mode, `atomic_f32`, `no_mask` or `mask_bottom_right`, and `seqlen_q == seqlen_k` with `seqlen_k % 128 == 0`.


## the asm kernel performance of the attention forwards and attention backwards.
the performance data was tested under the conditions of BF16 and BSHD in batch mode.

The MI355X numbers were re-measured on a gfx950 (MI355X) node with the asm-only
builds (`bash build_mha.sh fwd_v3` / `bash build_mha.sh bwd_v3`), taking the best
of 3 runs per cell:
```
    ./fwd.exe -prec=bf16 -iperm=0 -operm=0 -mode=0 -v=0 -lse=1 -fwd_v3=1        -b=$b -h=$hq -h_k=$hkv -s=$s -d=128 -mask=$causal
    ./bwd.exe -prec=bf16 -iperm=0 -operm=0 -mode=0 -v=0 -bwd_v3=1 -v3_bf16_cvt=1 -v3_atomic_fp32=0|1 -b=$b -h=$hq -h_k=$hkv -s=$s -d=128 -mask=$causal
```

![causal-fwd-perf picture](images/causal-fwd-perf.png)
![non-causal-fwd-perf picture](images/non-causal-fwd-perf.png)
*Figure 1: Evaluating GQA attention forwards performance under the conditions of batch=8, q_nheads=64 and kv_nheads=8.*

![causal-bwd-perf picture](images/causal-bwd-perf.png)
![non-causal-bwd-perf picture](images/non-causal-bwd-perf.png)
*Figure 2: Evaluating GQA attention backwards(a16) performance under the conditions of batch=8, q_nheads=64 and kv_nheads=8.*

**More performance test results are shown in the table below:**

| batch | q_nheads | kv_nheads | seqlen_q | seqlen_kv | hdim | causal | FWD(TFLOPS) |          | BWD-a16(TFLOPS) |          | BWD-a32(TFLOPS) |          |
|-------|----------|-----------|----------|-----------|------|--------|-------------|----------|-----------------|----------|-----------------|----------|
|       |          |           |          |           |      |        | MI300X      | MI355X   | MI300X          | MI355X   | MI300X          | MI355X   |
| 1     | 32       | 8         | 1024     | 1024      | 128  | 0      | 338.07      | 618.08   | 344.03          | 527.24   | 313.67          | 505.96   |
| 1     | 32       | 8         | 2048     | 2048      | 128  | 0      | 513.45      | 1131     | 311.9           | 919.32   | 269.19          | 707.47   |
| 1     | 32       | 8         | 4096     | 4096      | 128  | 0      | 527.73      | 1165.81  | 472.01          | 1066.46  | 423.53          | 789.01   |
| 1     | 32       | 8         | 8192     | 8192      | 128  | 0      | 558.17      | 1345.15  | 524.15          | 1195.55  | 481.28          | 822.25   |
| 1     | 32       | 8         | 10240    | 10240     | 128  | 0      | 549.73      | 1368.78  | 536.48          | 1192.5   | 491.28          | 830.41   |
| 4     | 32       | 8         | 1024     | 1024      | 128  | 0      | 458.41      | 1011.71  | 390.4           | 832.21   | 353.44          | 669.55   |
| 4     | 32       | 8         | 2048     | 2048      | 128  | 0      | 504.8       | 1052.48  | 459.52          | 985.15   | 430.81          | 745.72   |
| 4     | 32       | 8         | 4096     | 4096      | 128  | 0      | 577.16      | 1303.75  | 505.82          | 1143.05  | 457.38          | 804.06   |
| 4     | 32       | 8         | 8192     | 8192      | 128  | 0      | 574.62      | 1391.74  | 491.07          | 1207.72  | 458.72          | 831.5    |
| 4     | 32       | 8         | 10240    | 10240     | 128  | 0      | 584.66      | 1365.63  | 535.92          | 1216.57  | 476.64          | 840.8    |
| 8     | 32       | 8         | 1024     | 1024      | 128  | 0      | 459.43      | 1000.55  | 379.88          | 817.38   | 329.69          | 665.39   |
| 8     | 32       | 8         | 2048     | 2048      | 128  | 0      | 543.77      | 1130.99  | 475.12          | 1040.38  | 426.56          | 758.79   |
| 8     | 32       | 8         | 4096     | 4096      | 128  | 0      | 567.82      | 1339.02  | 519.34          | 1157.42  | 460.44          | 813.3    |
| 8     | 32       | 8         | 8192     | 8192      | 128  | 0      | 585.29      | 1358.5   | 518.07          | 1207.14  | 475.56          | 835.84   |
| 8     | 32       | 8         | 10240    | 10240     | 128  | 0      | 577.5       | 1369.63  | 534.98          | 1215.84  | 480.87          | 842.14   |
| 1     | 64       | 8         | 1024     | 1024      | 128  | 0      | 418.36      | 979.25   | 292.68          | 877.53   | 266.06          | 656.64   |
| 1     | 64       | 8         | 2048     | 2048      | 128  | 0      | 485.45      | 1153.95  | 437.26          | 915.31   | 393.6           | 720.28   |
| 1     | 64       | 8         | 4096     | 4096      | 128  | 0      | 546.34      | 1217.35  | 524.33          | 1121.22  | 470.15          | 794.69   |
| 1     | 64       | 8         | 8192     | 8192      | 128  | 0      | 591.37      | 1367.23  | 473             | 1209.98  | 441.82          | 826.49   |
| 1     | 64       | 8         | 10240    | 10240     | 128  | 0      | 572.09      | 1395.11  | 503.78          | 1190.89  | 460             | 834.74   |
| 4     | 64       | 8         | 1024     | 1024      | 128  | 0      | 440.07      | 1017.65  | 376.75          | 812.18   | 340.25          | 662.11   |
| 4     | 64       | 8         | 2048     | 2048      | 128  | 0      | 554.8       | 1141.57  | 477.46          | 1039.95  | 425.74          | 757.3    |
| 4     | 64       | 8         | 4096     | 4096      | 128  | 0      | 573.6       | 1334.37  | 510.76          | 1161.21  | 456.78          | 812.35   |
| 4     | 64       | 8         | 8192     | 8192      | 128  | 0      | 592.16      | 1354.33  | 511.65          | 1207.3   | 468.71          | 836.09   |
| 4     | 64       | 8         | 10240    | 10240     | 128  | 0      | 578.93      | 1363.39  | 535.75          | 1215.79  | 479.52          | 840.18   |
| 8     | 64       | 8         | 1024     | 1024      | 128  | 0      | 466.21      | 930.43   | 389.97          | 896.87   | 357.82          | 674.9    |
| 8     | 64       | 8         | 2048     | 2048      | 128  | 0      | 556.35      | 1222.56  | 479.74          | 1060.47  | 430.07          | 768.33   |
| 8     | 64       | 8         | 4096     | 4096      | 128  | 0      | 578.99      | 1341.97  | 482.86          | 1133.4   | 445.73          | 816.24   |
| 8     | 64       | 8         | 8192     | 8192      | 128  | 0      | 577.45      | 1364.85  | 537.04          | 1205.71  | 475.07          | 836.31   |
| 8     | 64       | 8         | 10240    | 10240     | 128  | 0      | 571.39      | 1383.15  | 550.19          | 1210.75  | 480.35          | 845.6    |
| 1     | 64       | 4         | 1024     | 1024      | 128  | 0      | 383.85      | 989.17   | 291.27          | 882.18   | 264.63          | 651.76   |
| 1     | 64       | 4         | 2048     | 2048      | 128  | 0      | 506.89      | 1138.65  | 443.31          | 919.74   | 396.33          | 729.71   |
| 1     | 64       | 4         | 4096     | 4096      | 128  | 0      | 549.2       | 1227.81  | 520.99          | 1127.74  | 467.24          | 794.29   |
| 1     | 64       | 4         | 8192     | 8192      | 128  | 0      | 591.77      | 1376.01  | 465.87          | 1208.58  | 439.94          | 826.85   |
| 1     | 64       | 4         | 10240    | 10240     | 128  | 0      | 571.59      | 1391.96  | 505.49          | 1220.42  | 459.64          | 836.4    |
| 4     | 64       | 4         | 1024     | 1024      | 128  | 0      | 460.34      | 1011.41  | 395.21          | 820.62   | 332.54          | 663.53   |
| 4     | 64       | 4         | 2048     | 2048      | 128  | 0      | 556.35      | 1158.81  | 474.83          | 1033.21  | 424.12          | 756.71   |
| 4     | 64       | 4         | 4096     | 4096      | 128  | 0      | 575.69      | 1334.24  | 519.08          | 1158.91  | 457.51          | 811.16   |
| 4     | 64       | 4         | 8192     | 8192      | 128  | 0      | 590.93      | 1352.78  | 513.66          | 1206.87  | 469.72          | 836.53   |
| 4     | 64       | 4         | 10240    | 10240     | 128  | 0      | 582.64      | 1361.86  | 534.39          | 1214.5   | 475.49          | 841.71   |
| 8     | 64       | 4         | 1024     | 1024      | 128  | 0      | 497.15      | 968.58   | 389.54          | 885.91   | 360.39          | 683.27   |
| 8     | 64       | 4         | 2048     | 2048      | 128  | 0      | 556.22      | 1221.03  | 478.01          | 1062.85  | 426.77          | 768.9    |
| 8     | 64       | 4         | 4096     | 4096      | 128  | 0      | 581.34      | 1353.66  | 481.35          | 1161.45  | 438.77          | 815.89   |
| 8     | 64       | 4         | 8192     | 8192      | 128  | 0      | 583.23      | 1363.81  | 536.72          | 1206.4   | 475.68          | 836.55   |
| 8     | 64       | 4         | 10240    | 10240     | 128  | 0      | 566.17      | 1378.5   | 550.05          | 1208.77  | 478.88          | 845.86   |
| 1     | 64       | 8         | 16384    | 16384     | 128  | 0      | 547.78      | 1358.82  | 519.21          | 1233.04  | 441.55          | 844.28   |
| 1     | 64       | 4         | 16384    | 16384     | 128  | 0      | 549.09      | 1357.12  | 516.26          | 1211.63  | 448.83          | 844.39   |
| 1     | 32       | 8         | 1024     | 1024      | 128  | 1      | 130.62      | 234.47   | 177.565         | 216.09   | 166.78          | 208.185  |
| 1     | 32       | 8         | 2048     | 2048      | 128  | 1      | 255.105     | 578.1    | 317.3           | 513.72   | 295.865         | 471.98   |
| 1     | 32       | 8         | 4096     | 4096      | 128  | 1      | 467.805     | 1104.96  | 317.685         | 869.19   | 296.025         | 714.5    |
| 1     | 32       | 8         | 8192     | 8192      | 128  | 1      | 522.68      | 1180.47  | 436.13          | 1072.28  | 388.235         | 777.76   |
| 1     | 32       | 8         | 10240    | 10240     | 128  | 1      | 440.12      | 1177.77  | 513.85          | 1037.73  | 244.705         | 765.455  |
| 4     | 32       | 8         | 1024     | 1024      | 128  | 1      | 334.005     | 707.4    | 257.115         | 587      | 226.39          | 489.62   |
| 4     | 32       | 8         | 2048     | 2048      | 128  | 1      | 419.435     | 850.43   | 377.51          | 751.69   | 330.23          | 599.415  |
| 4     | 32       | 8         | 4096     | 4096      | 128  | 1      | 486.73      | 1071.33  | 464.83          | 970.28   | 416.54          | 727.925  |
| 4     | 32       | 8         | 8192     | 8192      | 128  | 1      | 547.09      | 1302.93  | 468.205         | 1101.185 | 422.835         | 780.87   |
| 4     | 32       | 8         | 10240    | 10240     | 128  | 1      | 527.705     | 1319.19  | 474.205         | 1127.54  | 432.545         | 804.78   |
| 8     | 32       | 8         | 1024     | 1024      | 128  | 1      | 311.385     | 718.67   | 301.495         | 542.505  | 258.26          | 461.085  |
| 8     | 32       | 8         | 2048     | 2048      | 128  | 1      | 412.99      | 829.17   | 374.255         | 806.615  | 326.355         | 624.69   |
| 8     | 32       | 8         | 4096     | 4096      | 128  | 1      | 513.1       | 1141.34  | 454.36          | 996.29   | 409.05          | 734.075  |
| 8     | 32       | 8         | 8192     | 8192      | 128  | 1      | 537.36      | 1305.2   | 491.78          | 1104.53  | 441.4           | 785.89   |
| 8     | 32       | 8         | 10240    | 10240     | 128  | 1      | 556.045     | 1321.17  | 495.15          | 1128.37  | 443.78          | 802.775  |
| 1     | 64       | 8         | 1024     | 1024      | 128  | 1      | 228.54      | 426.39   | 283.58          | 390.355  | 242.43          | 374.255  |
| 1     | 64       | 8         | 2048     | 2048      | 128  | 1      | 392.425     | 940.54   | 279.72          | 707.55   | 257.855         | 607.725  |
| 1     | 64       | 8         | 4096     | 4096      | 128  | 1      | 474.385     | 986.72   | 420.265         | 946.565  | 378.155         | 713.03   |
| 1     | 64       | 8         | 8192     | 8192      | 128  | 1      | 518.29      | 1261.28  | 481.895         | 1091.225 | 433.285         | 773.91   |
| 1     | 64       | 8         | 10240    | 10240     | 128  | 1      | 510.895     | 1268.63  | 501.055         | 1123.16  | 447.995         | 792.56   |
| 4     | 64       | 8         | 1024     | 1024      | 128  | 1      | 326.51      | 705.51   | 311.005         | 546      | 266.9           | 463.625  |
| 4     | 64       | 8         | 2048     | 2048      | 128  | 1      | 425.735     | 819.54   | 377.225         | 778.245  | 326.805         | 626.28   |
| 4     | 64       | 8         | 4096     | 4096      | 128  | 1      | 513.79      | 1158.88  | 449             | 998.1    | 391.235         | 731.07   |
| 4     | 64       | 8         | 8192     | 8192      | 128  | 1      | 540.515     | 1306.37  | 482.505         | 1104.505 | 434.645         | 782.055  |
| 4     | 64       | 8         | 10240    | 10240     | 128  | 1      | 557.475     | 1321.14  | 493.745         | 1128.05  | 442.51          | 797.255  |
| 8     | 64       | 8         | 1024     | 1024      | 128  | 1      | 321.865     | 637.33   | 324.22          | 591.815  | 265.08          | 482.29   |
| 8     | 64       | 8         | 2048     | 2048      | 128  | 1      | 452.03      | 911.56   | 382.1           | 840.03   | 347.89          | 640.345  |
| 8     | 64       | 8         | 4096     | 4096      | 128  | 1      | 509.255     | 1187.64  | 457.05          | 1009.94  | 402.18          | 733.31   |
| 8     | 64       | 8         | 8192     | 8192      | 128  | 1      | 550.67      | 1304.18  | 474.02          | 1103.63  | 432.715         | 784.87   |
| 8     | 64       | 8         | 10240    | 10240     | 128  | 1      | 547.05      | 1332.9   | 489.075         | 1125.09  | 439.785         | 806.685  |
| 1     | 64       | 4         | 1024     | 1024      | 128  | 1      | 229.09      | 425.12   | 265.11          | 393.255  | 238.755         | 378.355  |
| 1     | 64       | 4         | 2048     | 2048      | 128  | 1      | 407.525     | 946.76   | 277.86          | 724.535  | 254.375         | 607.995  |
| 1     | 64       | 4         | 4096     | 4096      | 128  | 1      | 476.26      | 990.59   | 418.73          | 940.835  | 384.585         | 709.135  |
| 1     | 64       | 4         | 8192     | 8192      | 128  | 1      | 519.32      | 1271.33  | 480.06          | 1091.47  | 442.955         | 773.415  |
| 1     | 64       | 4         | 10240    | 10240     | 128  | 1      | 515.275     | 1319.44  | 499.72          | 1122.01  | 459.745         | 794.215  |
| 4     | 64       | 4         | 1024     | 1024      | 128  | 1      | 314.82      | 728.86   | 324.22          | 556.94   | 264.795         | 467.965  |
| 4     | 64       | 4         | 2048     | 2048      | 128  | 1      | 426.77      | 856.4    | 374.96          | 807.005  | 331.95          | 623.22   |
| 4     | 64       | 4         | 4096     | 4096      | 128  | 1      | 524.585     | 1166.02  | 453.97          | 996.42   | 405.02          | 728.495  |
| 4     | 64       | 4         | 8192     | 8192      | 128  | 1      | 540.935     | 1302.82  | 478.735         | 1100.705 | 430.95          | 780.27   |
| 4     | 64       | 4         | 10240    | 10240     | 128  | 1      | 560.63      | 1341.75  | 491.435         | 1127.945 | 441.345         | 800.555  |
| 8     | 64       | 4         | 1024     | 1024      | 128  | 1      | 348.76      | 624.18   | 315.035         | 587.465  | 267.48          | 488.92   |
| 8     | 64       | 4         | 2048     | 2048      | 128  | 1      | 461.89      | 935.33   | 400.31          | 843.345  | 352.7           | 640.085  |
| 8     | 64       | 4         | 4096     | 4096      | 128  | 1      | 513.795     | 1195.46  | 456.415         | 1010.345 | 402.68          | 732.155  |
| 8     | 64       | 4         | 8192     | 8192      | 128  | 1      | 552.78      | 1314.78  | 473.41          | 1104.48  | 434.51          | 783.295  |
| 8     | 64       | 4         | 10240    | 10240     | 128  | 1      | 548.65      | 1340.64  | 488.145         | 1124.74  | 435.745         | 804.075  |
| 1     | 64       | 8         | 16384    | 16384     | 128  | 1      | 541.55      | 1369.28  | 458.075         | 1158.875 | 412.04          | 814.065  |
| 1     | 64       | 4         | 16384    | 16384     | 128  | 1      | 544.1       | 1367.99  | 458.065         | 1158.905 | 419.975         | 816.12   |

