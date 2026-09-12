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


## the asm and opus kernel performance of the attention forwards and attention backwards.
the performance data was tested under the conditions of BF16 and BSHD in batch mode.

The table covers both head-dim pairs the asm forward supports, `hdim_q`/`hdim_v`
of 128/128 and 192/128, and carries three forward numbers per row: the asm kernel
on MI300X and on MI355X, plus the opus kernel (`-fwd_v3=2`) on MI355X. Every cell
is the best of 3 runs, measured with the asm-only builds (`bash build_mha.sh
fwd_v3` / `bash build_mha.sh bwd_v3`):
```
    ./fwd.exe -prec=bf16 -iperm=0 -operm=0 -mode=0 -v=0 -warmup=20 -repeat=50 -lse=1 -fwd_v3=1        -b=$b -h=$hq -h_k=$hkv -s=$s -d=$dq -d_v=$dv -mask=$causal
    ./fwd.exe -prec=bf16 -iperm=0 -operm=0 -mode=0 -v=0 -warmup=20 -repeat=50 -lse=1 -fwd_v3=2        -b=$b -h=$hq -h_k=$hkv -s=$s -d=$dq -d_v=$dv -mask=$causal
    ./bwd.exe -prec=bf16 -iperm=0 -operm=0 -mode=0 -v=0 -warmup=20 -repeat=50 -bwd_v3=1 -v3_bf16_cvt=1 -v3_atomic_fp32=0|1 -b=$b -h=$hq -h_k=$hkv -s=$s -d=$dq -d_v=$dv -mask=$causal
```

The `-warmup`/`-repeat` window matters and is part of the numbers above. At the
default 10/10 the short shapes finish before the clocks ramp and read low --
`b=4 h=32/8 s=1024` backward-a16 reports 753 TFLOPS at 10/10, 832 at 20/50 and
895 at 50/100, while shapes at `s>=8192` are already saturated and move by well
under 1%. Comparing a re-measurement against this table means matching 20/50.

`n/a` marks a cell that no kernel can fill rather than one that was skipped. The
opus kernels are compiled for gfx950 only, so they have no MI300X column at all,
and the 192/128 backward asm instances likewise only exist for gfx950, which
leaves the MI300X backward cells empty on those rows.

![causal-fwd-perf picture](images/causal-fwd-perf.png)
![non-causal-fwd-perf picture](images/non-causal-fwd-perf.png)
*Figure 1: Evaluating GQA attention forwards performance at hdim 128/128 under the conditions of batch=8, q_nheads=64 and kv_nheads=8. The third bar is the opus kernel, which exists on MI355X only.*

![causal-bwd-perf picture](images/causal-bwd-perf.png)
![non-causal-bwd-perf picture](images/non-causal-bwd-perf.png)
*Figure 2: Evaluating GQA attention backwards(a16) performance at hdim 128/128 under the conditions of batch=8, q_nheads=64 and kv_nheads=8.*

![causal-fwd-perf-dim-192_128 picture](images/causal-fwd-perf-dim-192_128.png)
![non-causal-fwd-perf-dim-192_128 picture](images/non-causal-fwd-perf-dim-192_128.png)
*Figure 3: The same forwards comparison at hdim 192/128 (batch=8, q_nheads=64, kv_nheads=8). The opus kernel leads the asm one by a wider margin here than at 128/128, and by more still under a causal mask.*

![causal-bwd-perf-dim-192_128 picture](images/causal-bwd-perf-dim-192_128.png)
![non-causal-bwd-perf-dim-192_128 picture](images/non-causal-bwd-perf-dim-192_128.png)
*Figure 4: GQA attention backwards(a16) performance at hdim 192/128 (batch=8, q_nheads=64, kv_nheads=8). Only MI355X appears: the 192/128 backward asm instances are built for gfx950 only, so there is no MI300X series to plot.*

**More performance test results are shown in the table below:**

| batch | q_nheads | kv_nheads | seqlen_q | seqlen_kv | hdim_q | hdim_v | causal | FWD(TFLOPS) |         |             | BWD-a16(TFLOPS) |         | BWD-a32(TFLOPS) |        |
|-------|----------|-----------|----------|-----------|--------|--------|--------|-------------|---------|-------------|-----------------|---------|-----------------|--------|
|       |          |           |          |           |        |        |        | MI300X      | MI355X  | MI355X-opus | MI300X          | MI355X  | MI300X          | MI355X |
| 1     | 32       | 8         | 1024     | 1024      | 128    | 128    | 0      | 338.07      | 611.33  | 634.41      | 344.03          | 524.45  | 313.67          | 495.3  |
| 1     | 32       | 8         | 2048     | 2048      | 128    | 128    | 0      | 513.45      | 979.59  | 1080.13     | 311.9           | 888.82  | 269.19          | 713.67 |
| 1     | 32       | 8         | 4096     | 4096      | 128    | 128    | 0      | 527.73      | 1165.96 | 1246.35     | 472.01          | 1090.56 | 423.53          | 790.13 |
| 1     | 32       | 8         | 8192     | 8192      | 128    | 128    | 0      | 558.17      | 1335.79 | 1428.02     | 524.15          | 1160.38 | 481.28          | 820.85 |
| 1     | 32       | 8         | 10240    | 10240     | 128    | 128    | 0      | 549.73      | 1279.7  | 1371.93     | 536.48          | 1141.88 | 491.28          | 829.2  |
| 4     | 32       | 8         | 1024     | 1024      | 128    | 128    | 0      | 458.41      | 933.47  | 938.35      | 390.4           | 842.84  | 353.44          | 677.38 |
| 4     | 32       | 8         | 2048     | 2048      | 128    | 128    | 0      | 504.8       | 1114.85 | 1177.98     | 459.52          | 1013.22 | 430.81          | 749.07 |
| 4     | 32       | 8         | 4096     | 4096      | 128    | 128    | 0      | 577.16      | 1304.49 | 1378.67     | 505.82          | 1103.02 | 457.38          | 804.14 |
| 4     | 32       | 8         | 8192     | 8192      | 128    | 128    | 0      | 574.62      | 1368.77 | 1466.34     | 491.07          | 1155.98 | 458.72          | 832.27 |
| 4     | 32       | 8         | 10240    | 10240     | 128    | 128    | 0      | 584.66      | 1335.39 | 1434.56     | 535.92          | 1110.94 | 476.64          | 835.11 |
| 8     | 32       | 8         | 1024     | 1024      | 128    | 128    | 0      | 459.43      | 895.95  | 954.28      | 379.88          | 858.25  | 329.69          | 674.34 |
| 8     | 32       | 8         | 2048     | 2048      | 128    | 128    | 0      | 543.77      | 1180.24 | 1241.23     | 475.12          | 1019.64 | 426.56          | 759.36 |
| 8     | 32       | 8         | 4096     | 4096      | 128    | 128    | 0      | 567.82      | 1272.19 | 1339.52     | 519.34          | 1082.25 | 460.44          | 812.12 |
| 8     | 32       | 8         | 8192     | 8192      | 128    | 128    | 0      | 585.29      | 1273.49 | 1426.85     | 518.07          | 1120.02 | 475.56          | 819.36 |
| 8     | 32       | 8         | 10240    | 10240     | 128    | 128    | 0      | 577.5       | 1308.18 | 1389.96     | 534.98          | 1127.22 | 480.87          | 839.77 |
| 1     | 64       | 8         | 1024     | 1024      | 128    | 128    | 0      | 418.36      | 937.64  | 998.54      | 292.68          | 733.67  | 266.06          | 647.25 |
| 1     | 64       | 8         | 2048     | 2048      | 128    | 128    | 0      | 485.45      | 1015.05 | 1087.95     | 437.26          | 990.8   | 393.6           | 738.25 |
| 1     | 64       | 8         | 4096     | 4096      | 128    | 128    | 0      | 546.34      | 1266.38 | 1345.33     | 524.33          | 1107.84 | 470.15          | 793.78 |
| 1     | 64       | 8         | 8192     | 8192      | 128    | 128    | 0      | 591.37      | 1308.55 | 1404.56     | 473             | 1147.31 | 441.82          | 827.4  |
| 1     | 64       | 8         | 10240    | 10240     | 128    | 128    | 0      | 572.09      | 1341.71 | 1446.53     | 503.78          | 1162.54 | 460             | 834.45 |
| 4     | 64       | 8         | 1024     | 1024      | 128    | 128    | 0      | 440.07      | 906.96  | 968.16      | 376.75          | 848.91  | 340.25          | 668.73 |
| 4     | 64       | 8         | 2048     | 2048      | 128    | 128    | 0      | 554.8       | 1199.52 | 1258.48     | 477.46          | 1029.28 | 425.74          | 760.33 |
| 4     | 64       | 8         | 4096     | 4096      | 128    | 128    | 0      | 573.6       | 1284.33 | 1363.8      | 510.76          | 1111.33 | 456.78          | 788.54 |
| 4     | 64       | 8         | 8192     | 8192      | 128    | 128    | 0      | 592.16      | 1319.8  | 1419.09     | 511.65          | 1123.81 | 468.71          | 834.79 |
| 4     | 64       | 8         | 10240    | 10240     | 128    | 128    | 0      | 578.93      | 1369.9  | 1476.41     | 535.75          | 1143.68 | 479.52          | 837    |
| 8     | 64       | 8         | 1024     | 1024      | 128    | 128    | 0      | 466.21      | 1000.24 | 1024.03     | 389.97          | 887.11  | 357.82          | 692.39 |
| 8     | 64       | 8         | 2048     | 2048      | 128    | 128    | 0      | 556.35      | 1236.94 | 1294.75     | 479.74          | 1024.43 | 430.07          | 766.46 |
| 8     | 64       | 8         | 4096     | 4096      | 128    | 128    | 0      | 578.99      | 1332.22 | 1410.67     | 482.86          | 1117.72 | 445.73          | 814.06 |
| 8     | 64       | 8         | 8192     | 8192      | 128    | 128    | 0      | 577.45      | 1307.16 | 1398.82     | 537.04          | 1095.65 | 475.07          | 828.14 |
| 8     | 64       | 8         | 10240    | 10240     | 128    | 128    | 0      | 571.39      | 1328.8  | 1425.87     | 550.19          | 1120.34 | 480.35          | 832.98 |
| 1     | 64       | 4         | 1024     | 1024      | 128    | 128    | 0      | 383.85      | 955.15  | 982.26      | 291.27          | 755.36  | 264.63          | 641.44 |
| 1     | 64       | 4         | 2048     | 2048      | 128    | 128    | 0      | 506.89      | 1022.1  | 1096.52     | 443.31          | 978.62  | 396.33          | 737.1  |
| 1     | 64       | 4         | 4096     | 4096      | 128    | 128    | 0      | 549.2       | 1291.01 | 1360.18     | 520.99          | 1121.82 | 467.24          | 793.6  |
| 1     | 64       | 4         | 8192     | 8192      | 128    | 128    | 0      | 591.77      | 1314.78 | 1412.08     | 465.87          | 1138.34 | 439.94          | 826.9  |
| 1     | 64       | 4         | 10240    | 10240     | 128    | 128    | 0      | 571.59      | 1311.61 | 1410.35     | 505.49          | 1143.14 | 459.64          | 834.59 |
| 4     | 64       | 4         | 1024     | 1024      | 128    | 128    | 0      | 460.34      | 991.29  | 969.79      | 395.21          | 863.53  | 332.54          | 671.79 |
| 4     | 64       | 4         | 2048     | 2048      | 128    | 128    | 0      | 556.35      | 1174.7  | 1229.96     | 474.83          | 1001.23 | 424.12          | 758.72 |
| 4     | 64       | 4         | 4096     | 4096      | 128    | 128    | 0      | 575.69      | 1297.13 | 1385.95     | 519.08          | 1099.29 | 457.51          | 776.98 |
| 4     | 64       | 4         | 8192     | 8192      | 128    | 128    | 0      | 590.93      | 1263.93 | 1403.43     | 513.66          | 1119.44 | 469.72          | 827.39 |
| 4     | 64       | 4         | 10240    | 10240     | 128    | 128    | 0      | 582.64      | 1316    | 1416.05     | 534.39          | 1103    | 475.49          | 826.01 |
| 8     | 64       | 4         | 1024     | 1024      | 128    | 128    | 0      | 497.15      | 1029.66 | 1034.15     | 389.54          | 905.19  | 360.39          | 687.16 |
| 8     | 64       | 4         | 2048     | 2048      | 128    | 128    | 0      | 556.22      | 1235.04 | 1297.66     | 478.01          | 1032.93 | 426.77          | 765.75 |
| 8     | 64       | 4         | 4096     | 4096      | 128    | 128    | 0      | 581.34      | 1298.15 | 1381.05     | 481.35          | 1087.85 | 438.77          | 800.23 |
| 8     | 64       | 4         | 8192     | 8192      | 128    | 128    | 0      | 583.23      | 1341.3  | 1427.16     | 536.72          | 1145.34 | 475.68          | 832.28 |
| 8     | 64       | 4         | 10240    | 10240     | 128    | 128    | 0      | 566.17      | 1295.94 | 1397.4      | 550.05          | 1115.84 | 478.88          | 836.82 |
| 1     | 64       | 8         | 16384    | 16384     | 128    | 128    | 0      | 547.78      | 1315.4  | 1442.32     | 519.21          | 1156.17 | 441.55          | 843.58 |
| 1     | 64       | 4         | 16384    | 16384     | 128    | 128    | 0      | 549.09      | 1313.86 | 1418.75     | 516.26          | 1115.12 | 448.83          | 828.44 |
| 1     | 32       | 8         | 1024     | 1024      | 128    | 128    | 1      | 130.62      | 236.56  | 329.04      | 177.565         | 216.49  | 166.78          | 208.52 |
| 1     | 32       | 8         | 2048     | 2048      | 128    | 128    | 1      | 255.105     | 564.78  | 703.19      | 317.3           | 518.11  | 295.865         | 489.09 |
| 1     | 32       | 8         | 4096     | 4096      | 128    | 128    | 1      | 467.805     | 966.7   | 926.29      | 317.685         | 926.41  | 296.025         | 718.1  |
| 1     | 32       | 8         | 8192     | 8192      | 128    | 128    | 1      | 522.68      | 1230.26 | 1208.37     | 436.13          | 1056.03 | 388.235         | 776.05 |
| 1     | 32       | 8         | 10240    | 10240     | 128    | 128    | 1      | 440.12      | 1180.57 | 1274.21     | 513.85          | 962.88  | 244.705         | 757.05 |
| 4     | 32       | 8         | 1024     | 1024      | 128    | 128    | 1      | 334.005     | 617.33  | 561.53      | 257.115         | 550.7   | 226.39          | 476.25 |
| 4     | 32       | 8         | 2048     | 2048      | 128    | 128    | 1      | 419.435     | 837.37  | 858.44      | 377.51          | 791.47  | 330.23          | 603.59 |
| 4     | 32       | 8         | 4096     | 4096      | 128    | 128    | 1      | 486.73      | 1109.98 | 1142.41     | 464.83          | 971.76  | 416.54          | 725.35 |
| 4     | 32       | 8         | 8192     | 8192      | 128    | 128    | 1      | 547.09      | 1283.77 | 1323.5      | 468.205         | 1046.54 | 422.835         | 780.1  |
| 4     | 32       | 8         | 10240    | 10240     | 128    | 128    | 1      | 527.705     | 1324.17 | 1383.25     | 474.205         | 1078.03 | 432.545         | 780.1  |
| 8     | 32       | 8         | 1024     | 1024      | 128    | 128    | 1      | 311.385     | 623.13  | 630.6       | 301.495         | 580.56  | 258.26          | 468.91 |
| 8     | 32       | 8         | 2048     | 2048      | 128    | 128    | 1      | 412.99      | 892.39  | 912.72      | 374.255         | 809.66  | 326.355         | 625.18 |
| 8     | 32       | 8         | 4096     | 4096      | 128    | 128    | 1      | 513.1       | 1138.14 | 1205.94     | 454.36          | 958.54  | 409.05          | 733.74 |
| 8     | 32       | 8         | 8192     | 8192      | 128    | 128    | 1      | 537.36      | 1242.64 | 1301.74     | 491.78          | 1044.62 | 441.4           | 782.27 |
| 8     | 32       | 8         | 10240    | 10240     | 128    | 128    | 1      | 556.045     | 1284.59 | 1356.5      | 495.15          | 1071.6  | 443.78          | 799.87 |
| 1     | 64       | 8         | 1024     | 1024      | 128    | 128    | 1      | 228.54      | 426.91  | 578.13      | 283.58          | 389.99  | 242.43          | 368.38 |
| 1     | 64       | 8         | 2048     | 2048      | 128    | 128    | 1      | 392.425     | 808.75  | 771.91      | 279.72          | 764.07  | 257.855         | 615.78 |
| 1     | 64       | 8         | 4096     | 4096      | 128    | 128    | 1      | 474.385     | 1071.22 | 1085.31     | 420.265         | 962.62  | 378.155         | 713.23 |
| 1     | 64       | 8         | 8192     | 8192      | 128    | 128    | 1      | 518.29      | 1262.01 | 1292.45     | 481.895         | 1055.45 | 433.285         | 771.66 |
| 1     | 64       | 8         | 10240    | 10240     | 128    | 128    | 1      | 510.895     | 1270.57 | 1310.38     | 501.055         | 1074.07 | 447.995         | 790.38 |
| 4     | 64       | 8         | 1024     | 1024      | 128    | 128    | 1      | 326.51      | 616.11  | 630.45      | 311.005         | 578.31  | 266.9           | 470.24 |
| 4     | 64       | 8         | 2048     | 2048      | 128    | 128    | 1      | 425.735     | 877.06  | 909.66      | 377.225         | 801.96  | 326.805         | 622.79 |
| 4     | 64       | 8         | 4096     | 4096      | 128    | 128    | 1      | 513.79      | 1152.32 | 1209.22     | 449             | 967.83  | 391.235         | 725.12 |
| 4     | 64       | 8         | 8192     | 8192      | 128    | 128    | 1      | 540.515     | 1248.33 | 1322.03     | 482.505         | 1033.69 | 434.645         | 781.88 |
| 4     | 64       | 8         | 10240    | 10240     | 128    | 128    | 1      | 557.475     | 1231.22 | 1375.62     | 493.745         | 1018.97 | 442.51          | 771.85 |
| 8     | 64       | 8         | 1024     | 1024      | 128    | 128    | 1      | 321.865     | 648.02  | 663.71      | 324.22          | 617.06  | 265.08          | 485.58 |
| 8     | 64       | 8         | 2048     | 2048      | 128    | 128    | 1      | 452.03      | 940.54  | 991.93      | 382.1           | 828.86  | 347.89          | 637.32 |
| 8     | 64       | 8         | 4096     | 4096      | 128    | 128    | 1      | 509.255     | 1148.56 | 1232.4      | 457.05          | 947.57  | 402.18          | 709.2  |
| 8     | 64       | 8         | 8192     | 8192      | 128    | 128    | 1      | 550.67      | 1269.79 | 1371.71     | 474.02          | 1060.17 | 432.715         | 774.93 |
| 8     | 64       | 8         | 10240    | 10240     | 128    | 128    | 1      | 547.05      | 1233.29 | 1350.51     | 489.075         | 1017.18 | 439.785         | 795.86 |
| 1     | 64       | 4         | 1024     | 1024      | 128    | 128    | 1      | 229.09      | 423     | 575.77      | 265.11          | 396.06  | 238.755         | 369.25 |
| 1     | 64       | 4         | 2048     | 2048      | 128    | 128    | 1      | 407.525     | 798.57  | 786.08      | 277.86          | 764.65  | 254.375         | 612.72 |
| 1     | 64       | 4         | 4096     | 4096      | 128    | 128    | 1      | 476.26      | 1078.5  | 1095.18     | 418.73          | 948.3   | 384.585         | 716.96 |
| 1     | 64       | 4         | 8192     | 8192      | 128    | 128    | 1      | 519.32      | 1289.36 | 1327.75     | 480.06          | 1062.7  | 442.955         | 767.41 |
| 1     | 64       | 4         | 10240    | 10240     | 128    | 128    | 1      | 515.275     | 1279.08 | 1321.25     | 499.72          | 1074.1  | 459.745         | 794.29 |
| 4     | 64       | 4         | 1024     | 1024      | 128    | 128    | 1      | 314.82      | 633.23  | 619.85      | 324.22          | 591.69  | 264.795         | 474.87 |
| 4     | 64       | 4         | 2048     | 2048      | 128    | 128    | 1      | 426.77      | 918.68  | 940.75      | 374.96          | 815.19  | 331.95          | 623.91 |
| 4     | 64       | 4         | 4096     | 4096      | 128    | 128    | 1      | 524.585     | 1131.42 | 1179.71     | 453.97          | 951.51  | 405.02          | 724.97 |
| 4     | 64       | 4         | 8192     | 8192      | 128    | 128    | 1      | 540.935     | 1260.24 | 1331.16     | 478.735         | 1046.21 | 430.95          | 778.6  |
| 4     | 64       | 4         | 10240    | 10240     | 128    | 128    | 1      | 560.63      | 1262.46 | 1286.6      | 491.435         | 1065.12 | 441.345         | 790.43 |
| 8     | 64       | 4         | 1024     | 1024      | 128    | 128    | 1      | 348.76      | 667.62  | 680.15      | 315.035         | 615.44  | 267.48          | 491.53 |
| 8     | 64       | 4         | 2048     | 2048      | 128    | 128    | 1      | 461.89      | 965.56  | 1004.68     | 400.31          | 845.29  | 352.7           | 637.62 |
| 8     | 64       | 4         | 4096     | 4096      | 128    | 128    | 1      | 513.795     | 1168.12 | 1235.48     | 456.415         | 952.51  | 402.68          | 730.9  |
| 8     | 64       | 4         | 8192     | 8192      | 128    | 128    | 1      | 552.78      | 1269.01 | 1348.23     | 473.41          | 1019.49 | 434.51          | 781.83 |
| 8     | 64       | 4         | 10240    | 10240     | 128    | 128    | 1      | 548.65      | 1293.74 | 1384.62     | 488.145         | 1073.01 | 435.745         | 800.86 |
| 1     | 64       | 8         | 16384    | 16384     | 128    | 128    | 1      | 541.55      | 1272.4  | 1326.56     | 458.075         | 1067.25 | 412.04          | 800.87 |
| 1     | 64       | 4         | 16384    | 16384     | 128    | 128    | 1      | 544.1       | 1299.18 | 1355.55     | 458.065         | 1098.61 | 419.975         | 816.11 |
| 1     | 32       | 8         | 1024     | 1024      | 192    | 128    | 0      | 375.85      | 908.19  | 666.43      | n/a             | 530.65  | n/a             | 397.33 |
| 1     | 32       | 8         | 2048     | 2048      | 192    | 128    | 0      | 482.74      | 951.88  | 1076.27     | n/a             | 710.94  | n/a             | 478.55 |
| 1     | 32       | 8         | 4096     | 4096      | 192    | 128    | 0      | 494.19      | 1227.79 | 1302.98     | n/a             | 954.34  | n/a             | 511.58 |
| 1     | 32       | 8         | 8192     | 8192      | 192    | 128    | 0      | 575.51      | 1318.3  | 1421.27     | n/a             | 940.08  | n/a             | 533.42 |
| 1     | 32       | 8         | 10240    | 10240     | 192    | 128    | 0      | 557.43      | 1314.18 | 1421.62     | n/a             | 994.79  | n/a             | 545.83 |
| 4     | 32       | 8         | 1024     | 1024      | 192    | 128    | 0      | 437.19      | 827.68  | 931.27      | n/a             | 709.67  | n/a             | 438.03 |
| 4     | 32       | 8         | 2048     | 2048      | 192    | 128    | 0      | 508.79      | 1103.5  | 1183.54     | n/a             | 869.89  | n/a             | 495.64 |
| 4     | 32       | 8         | 4096     | 4096      | 192    | 128    | 0      | 559.41      | 1304.38 | 1405.79     | n/a             | 963.68  | n/a             | 523.84 |
| 4     | 32       | 8         | 8192     | 8192      | 192    | 128    | 0      | 565.42      | 1303.6  | 1410.53     | n/a             | 1004.29 | n/a             | 547.19 |
| 4     | 32       | 8         | 10240    | 10240     | 192    | 128    | 0      | 558.48      | 1322.7  | 1442.36     | n/a             | 1010.66 | n/a             | 541.06 |
| 8     | 32       | 8         | 1024     | 1024      | 192    | 128    | 0      | 431.01      | 921.16  | 980.99      | n/a             | 744.15  | n/a             | 435.55 |
| 8     | 32       | 8         | 2048     | 2048      | 192    | 128    | 0      | 533.17      | 1183.48 | 1268.1      | n/a             | 916.35  | n/a             | 507.92 |
| 8     | 32       | 8         | 4096     | 4096      | 192    | 128    | 0      | 565.67      | 1300.56 | 1409.14     | n/a             | 951.7   | n/a             | 528.69 |
| 8     | 32       | 8         | 8192     | 8192      | 192    | 128    | 0      | 558.15      | 1340.98 | 1452.87     | n/a             | 1031.7  | n/a             | 546.42 |
| 8     | 32       | 8         | 10240    | 10240     | 192    | 128    | 0      | 571.3       | 1303.82 | 1415.89     | n/a             | 979.64  | n/a             | 543.65 |
| 1     | 64       | 8         | 1024     | 1024      | 192    | 128    | 0      | 401.92      | 844.75  | 984.73      | n/a             | 602.53  | n/a             | 426.48 |
| 1     | 64       | 8         | 2048     | 2048      | 192    | 128    | 0      | 434.98      | 1029.17 | 1094.46     | n/a             | 865.1   | n/a             | 497.47 |
| 1     | 64       | 8         | 4096     | 4096      | 192    | 128    | 0      | 547.17      | 1286.67 | 1381.96     | n/a             | 934.28  | n/a             | 511.19 |
| 1     | 64       | 8         | 8192     | 8192      | 192    | 128    | 0      | 557         | 1359.47 | 1462.92     | n/a             | 1014.68 | n/a             | 545    |
| 1     | 64       | 8         | 10240    | 10240     | 192    | 128    | 0      | 566.67      | 1329.09 | 1436.48     | n/a             | 988.56  | n/a             | 544.03 |
| 4     | 64       | 8         | 1024     | 1024      | 192    | 128    | 0      | 436.5       | 926.47  | 996.24      | n/a             | 739.56  | n/a             | 437.77 |
| 4     | 64       | 8         | 2048     | 2048      | 192    | 128    | 0      | 523.61      | 1200.32 | 1287.86     | n/a             | 914.63  | n/a             | 508.5  |
| 4     | 64       | 8         | 4096     | 4096      | 192    | 128    | 0      | 555.97      | 1281.7  | 1361.77     | n/a             | 934     | n/a             | 528.45 |
| 4     | 64       | 8         | 8192     | 8192      | 192    | 128    | 0      | 548.34      | 1322.16 | 1436.56     | n/a             | 993.37  | n/a             | 543.42 |
| 4     | 64       | 8         | 10240    | 10240     | 192    | 128    | 0      | 536.27      | 1283.59 | 1390.28     | n/a             | 1017.38 | n/a             | 542.76 |
| 8     | 64       | 8         | 1024     | 1024      | 192    | 128    | 0      | 460.59      | 986.39  | 1069.44     | n/a             | 756.23  | n/a             | 442.08 |
| 8     | 64       | 8         | 2048     | 2048      | 192    | 128    | 0      | 530.1       | 1225.99 | 1330.67     | n/a             | 924.37  | n/a             | 512.04 |
| 8     | 64       | 8         | 4096     | 4096      | 192    | 128    | 0      | 550.98      | 1247.5  | 1412.32     | n/a             | 946.92  | n/a             | 530.39 |
| 8     | 64       | 8         | 8192     | 8192      | 192    | 128    | 0      | 540.17      | 1329.64 | 1444.29     | n/a             | 1030.41 | n/a             | 549.02 |
| 8     | 64       | 8         | 10240    | 10240     | 192    | 128    | 0      | 530.87      | 1325.56 | 1458.79     | n/a             | 1022.24 | n/a             | 550.2  |
| 1     | 64       | 4         | 1024     | 1024      | 192    | 128    | 0      | 418.14      | 871.73  | 956.93      | n/a             | 585.12  | n/a             | 422.13 |
| 1     | 64       | 4         | 2048     | 2048      | 192    | 128    | 0      | 455.99      | 1047.59 | 1125.68     | n/a             | 858.55  | n/a             | 499.26 |
| 1     | 64       | 4         | 4096     | 4096      | 192    | 128    | 0      | 548.24      | 1264.54 | 1357.48     | n/a             | 947.29  | n/a             | 509.38 |
| 1     | 64       | 4         | 8192     | 8192      | 192    | 128    | 0      | 559.25      | 1331.8  | 1441.19     | n/a             | 917.09  | n/a             | 545.44 |
| 1     | 64       | 4         | 10240    | 10240     | 192    | 128    | 0      | 574.18      | 1371.57 | 1480.51     | n/a             | 980.52  | n/a             | 526.56 |
| 4     | 64       | 4         | 1024     | 1024      | 192    | 128    | 0      | 448.71      | 937.98  | 1004.23     | n/a             | 745.33  | n/a             | 437.46 |
| 4     | 64       | 4         | 2048     | 2048      | 192    | 128    | 0      | 531.22      | 1204.34 | 1295.76     | n/a             | 916.5   | n/a             | 507.11 |
| 4     | 64       | 4         | 4096     | 4096      | 192    | 128    | 0      | 565.29      | 1327.03 | 1430.08     | n/a             | 965.22  | n/a             | 528.47 |
| 4     | 64       | 4         | 8192     | 8192      | 192    | 128    | 0      | 559.48      | 1305.73 | 1417.79     | n/a             | 994.37  | n/a             | 549.15 |
| 4     | 64       | 4         | 10240    | 10240     | 192    | 128    | 0      | 572.1       | 1324.54 | 1406.62     | n/a             | 1015.16 | n/a             | 544.08 |
| 8     | 64       | 4         | 1024     | 1024      | 192    | 128    | 0      | 480.31      | 994.24  | 1072.83     | n/a             | 770.13  | n/a             | 442.9  |
| 8     | 64       | 4         | 2048     | 2048      | 192    | 128    | 0      | 544.32      | 1214.4  | 1332.15     | n/a             | 911.19  | n/a             | 513.59 |
| 8     | 64       | 4         | 4096     | 4096      | 192    | 128    | 0      | 555.52      | 1342.32 | 1359.51     | n/a             | 951.55  | n/a             | 526.31 |
| 8     | 64       | 4         | 8192     | 8192      | 192    | 128    | 0      | 572.77      | 1303.29 | 1427.79     | n/a             | 987.32  | n/a             | 543.16 |
| 8     | 64       | 4         | 10240    | 10240     | 192    | 128    | 0      | 578.38      | 1337.25 | 1454.98     | n/a             | 1017.69 | n/a             | 549.79 |
| 1     | 64       | 8         | 16384    | 16384     | 192    | 128    | 0      | 552.78      | 1352.18 | 1471.84     | n/a             | 1029.29 | n/a             | 549.87 |
| 1     | 64       | 4         | 16384    | 16384     | 192    | 128    | 0      | 570.23      | 1316.8  | 1432.91     | n/a             | 993.7   | n/a             | 548.03 |
| 1     | 32       | 8         | 1024     | 1024      | 192    | 128    | 1      | 251.14      | 345.46  | 344.02      | n/a             | 293.47  | n/a             | 278.55 |
| 1     | 32       | 8         | 2048     | 2048      | 192    | 128    | 1      | 391.96      | 683.2   | 622.14      | n/a             | 586.9   | n/a             | 436.23 |
| 1     | 32       | 8         | 4096     | 4096      | 192    | 128    | 1      | 453.39      | 868.37  | 1049.54     | n/a             | 740.67  | n/a             | 472.81 |
| 1     | 32       | 8         | 8192     | 8192      | 192    | 128    | 1      | 541.39      | 1100.04 | 1359.12     | n/a             | 928.05  | n/a             | 514.13 |
| 1     | 32       | 8         | 10240    | 10240     | 192    | 128    | 1      | 560.41      | 1101.36 | 1281.26     | n/a             | 868.07  | n/a             | 513.73 |
| 4     | 32       | 8         | 1024     | 1024      | 192    | 128    | 1      | 354.29      | 572.72  | 640.22      | n/a             | 496.25  | n/a             | 356.17 |
| 4     | 32       | 8         | 2048     | 2048      | 192    | 128    | 1      | 406.51      | 785.65  | 912.58      | n/a             | 735.33  | n/a             | 437.5  |
| 4     | 32       | 8         | 4096     | 4096      | 192    | 128    | 1      | 522.02      | 966.62  | 1176.36     | n/a             | 846.59  | n/a             | 487.22 |
| 4     | 32       | 8         | 8192     | 8192      | 192    | 128    | 1      | 550.77      | 1059.42 | 1367.73     | n/a             | 949.8   | n/a             | 529.51 |
| 4     | 32       | 8         | 10240    | 10240     | 192    | 128    | 1      | 551.88      | 1073.18 | 1360.7      | n/a             | 946.74  | n/a             | 528.65 |
| 8     | 32       | 8         | 1024     | 1024      | 192    | 128    | 1      | 348.14      | 601.98  | 653.15      | n/a             | 570.17  | n/a             | 370.7  |
| 8     | 32       | 8         | 2048     | 2048      | 192    | 128    | 1      | 430.64      | 841.63  | 974.47      | n/a             | 765.23  | n/a             | 446.2  |
| 8     | 32       | 8         | 4096     | 4096      | 192    | 128    | 1      | 523.1       | 997.9   | 1231.84     | n/a             | 895.55  | n/a             | 499.14 |
| 8     | 32       | 8         | 8192     | 8192      | 192    | 128    | 1      | 541.58      | 1096.95 | 1363.01     | n/a             | 925.16  | n/a             | 532.99 |
| 8     | 32       | 8         | 10240    | 10240     | 192    | 128    | 1      | 547.89      | 1034.75 | 1387.68     | n/a             | 980.54  | n/a             | 539.07 |
| 1     | 64       | 8         | 1024     | 1024      | 192    | 128    | 1      | 319         | 580.77  | 523.83      | n/a             | 451.47  | n/a             | 355.38 |
| 1     | 64       | 8         | 2048     | 2048      | 192    | 128    | 1      | 440.65      | 716.48  | 863.81      | n/a             | 634.99  | n/a             | 425.58 |
| 1     | 64       | 8         | 4096     | 4096      | 192    | 128    | 1      | 486.59      | 951.78  | 1159.65     | n/a             | 869.57  | n/a             | 488.44 |
| 1     | 64       | 8         | 8192     | 8192      | 192    | 128    | 1      | 553.13      | 1104    | 1373.05     | n/a             | 893.48  | n/a             | 513    |
| 1     | 64       | 8         | 10240    | 10240     | 192    | 128    | 1      | 559.31      | 1125.47 | 1414.66     | n/a             | 961.07  | n/a             | 534.24 |
| 4     | 64       | 8         | 1024     | 1024      | 192    | 128    | 1      | 358.89      | 593.29  | 673.38      | n/a             | 567.13  | n/a             | 367.44 |
| 4     | 64       | 8         | 2048     | 2048      | 192    | 128    | 1      | 431.39      | 831.62  | 984.91      | n/a             | 764.04  | n/a             | 447.26 |
| 4     | 64       | 8         | 4096     | 4096      | 192    | 128    | 1      | 518.09      | 1000.87 | 1244.2      | n/a             | 894.16  | n/a             | 501.41 |
| 4     | 64       | 8         | 8192     | 8192      | 192    | 128    | 1      | 539.21      | 1036.87 | 1339.3      | n/a             | 939.79  | n/a             | 532.42 |
| 4     | 64       | 8         | 10240    | 10240     | 192    | 128    | 1      | 540.14      | 1013.73 | 1386.31     | n/a             | 983.79  | n/a             | 534.73 |
| 8     | 64       | 8         | 1024     | 1024      | 192    | 128    | 1      | 350.34      | 633.48  | 715.94      | n/a             | 603.94  | n/a             | 373.33 |
| 8     | 64       | 8         | 2048     | 2048      | 192    | 128    | 1      | 469.64      | 878.83  | 1029.19     | n/a             | 761.98  | n/a             | 456.12 |
| 8     | 64       | 8         | 4096     | 4096      | 192    | 128    | 1      | 515.66      | 1031.79 | 1278.59     | n/a             | 895.58  | n/a             | 488.42 |
| 8     | 64       | 8         | 8192     | 8192      | 192    | 128    | 1      | 530.47      | 1036.98 | 1380.08     | n/a             | 966.13  | n/a             | 529.63 |
| 8     | 64       | 8         | 10240    | 10240     | 192    | 128    | 1      | 541.1       | 1012.44 | 1391.44     | n/a             | 968.87  | n/a             | 539.23 |
| 1     | 64       | 4         | 1024     | 1024      | 192    | 128    | 1      | 269.09      | 589.85  | 549.1       | n/a             | 471.57  | n/a             | 361.06 |
| 1     | 64       | 4         | 2048     | 2048      | 192    | 128    | 1      | 456.89      | 696.73  | 877.86      | n/a             | 630.33  | n/a             | 430.84 |
| 1     | 64       | 4         | 4096     | 4096      | 192    | 128    | 1      | 504.33      | 963.33  | 1188.56     | n/a             | 862.95  | n/a             | 485.59 |
| 1     | 64       | 4         | 8192     | 8192      | 192    | 128    | 1      | 561.86      | 1103.06 | 1334.63     | n/a             | 893.3   | n/a             | 513.52 |
| 1     | 64       | 4         | 10240    | 10240     | 192    | 128    | 1      | 566.33      | 1124.98 | 1390.27     | n/a             | 889.48  | n/a             | 527.5  |
| 4     | 64       | 4         | 1024     | 1024      | 192    | 128    | 1      | 367.64      | 599.87  | 712.05      | n/a             | 578.37  | n/a             | 363.61 |
| 4     | 64       | 4         | 2048     | 2048      | 192    | 128    | 1      | 444.97      | 834.92  | 988.52      | n/a             | 750.05  | n/a             | 445.79 |
| 4     | 64       | 4         | 4096     | 4096      | 192    | 128    | 1      | 531.91      | 1029.5  | 1274.78     | n/a             | 894.46  | n/a             | 501.77 |
| 4     | 64       | 4         | 8192     | 8192      | 192    | 128    | 1      | 550.59      | 1049.65 | 1397.8      | n/a             | 955.38  | n/a             | 532.03 |
| 4     | 64       | 4         | 10240    | 10240     | 192    | 128    | 1      | 553.92      | 1036.25 | 1368.65     | n/a             | 963.15  | n/a             | 535.53 |
| 8     | 64       | 4         | 1024     | 1024      | 192    | 128    | 1      | 359.01      | 638.06  | 736.25      | n/a             | 607.15  | n/a             | 371.74 |
| 8     | 64       | 4         | 2048     | 2048      | 192    | 128    | 1      | 479.86      | 892.4   | 1047.39     | n/a             | 763.04  | n/a             | 455.12 |
| 8     | 64       | 4         | 4096     | 4096      | 192    | 128    | 1      | 529.82      | 1033.87 | 1270.88     | n/a             | 888.66  | n/a             | 495.62 |
| 8     | 64       | 4         | 8192     | 8192      | 192    | 128    | 1      | 541.56      | 1083.96 | 1365.28     | n/a             | 961.04  | n/a             | 529.01 |
| 8     | 64       | 4         | 10240    | 10240     | 192    | 128    | 1      | 550.55      | 1038.2  | 1383.65     | n/a             | 965.25  | n/a             | 535.29 |
| 1     | 64       | 8         | 16384    | 16384     | 192    | 128    | 1      | 560.35      | 1143.26 | 1397.76     | n/a             | 993.58  | n/a             | 540.74 |
| 1     | 64       | 4         | 16384    | 16384     | 192    | 128    | 1      | 566.76      | 1087.91 | 1428.83     | n/a             | 990.67  | n/a             | 545.86 |

