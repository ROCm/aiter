# CKTILE gemm a8w8 bpreshuffle tune

1. Install aiter:  
`python3 setup.py develop`

2. Tune gemm a8w8: 
 First add GEMM shapes in `aiter/configs/a8w8_bpreshuffle_untuned_gemm.csv`, then run the following cmd to start tuning, please wait a few minutes as it will build gemm_a8w8_bpreshuffle_tune via jit:  
`FLATMM_HIP_CLANG_PATH=/data/llvm-project/build/bin/ python3 csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py -i aiter/configs/a8w8_bpreshuffle_untuned_gemm.csv -o aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv --libtype cktile`  
If you want to use split K kernels, you can add the `-k` parameter at the end, notice that should change `bias` to `bias/(2^k)`.
This will tune both ck and cktile implementations, if you want to tune cktile only, you can add the `--libtype cktile` parameter at the end.
You can find the results of the tuning in `aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv`.

3. Test the performance, modify the test instance in `op_tests/test_gemm_a8w8.py` and run it, please wait a few minutes as it will build gemm_a8w8_bpreshuffle_cktile kernels in `aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv` via jit：  
`FLATMM_HIP_CLANG_PATH=/data/llvm-project/build/bin/ python3 op_tests/test_gemm_a8w8.py`


## More
If you want to re-install gemm_a8w8_bpreshuffle_cktile, you should remove `aiter/jit/module_gemm_a8w8_bpreshuffle_cktile.so` and `aiter/jit/build/module_gemm_a8w8_bpreshuffle_cktile` first.
If you use flag `PREBUILD_KERNELS=1` when you install aiter, it will build gemm a8w8 kernels in `aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv` by default. If you want to use the new result of gemm_a8w8_bpreshuffle_cktile_tune, please remove `build` and `*.so` first, then re-install aiter after finishing tune.
