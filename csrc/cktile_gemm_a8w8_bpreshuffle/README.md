# CKTILE gemm a8w8 bpreshuffle tune

1. Install aiter:  
`python3 setup.py develop`

2. Tune gemm a8w8: 
 First add GEMM shapes in `aiter/configs/a8w8_bpreshuffle_cktile_untuned_gemm.csv`, then run the following cmd to start tuning, please wait a few minutes as it will build gemm_a8w8_bpreshuffle_cktile_tune via jit:  
`GEMM_CKTILE_BPRESHUFFLE_HIP_CLANG_PATH=/data/llvm-project/build/bin/ python3 csrc/cktile_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_cktile_tune.py -i aiter/configs/a8w8_bpreshuffle_cktile_untuned_gemm.csv -o aiter/configs/a8w8_bpreshuffle_cktile_tuned_gemm.csv`  
If you want to use split K kernels, you can add the `-k` parameter at the end, notice that should change `bias` to `bias/(2^k)`.
You can find the results of the tuning in `aiter/configs/a8w8_bpreshuffle_cktile_tuned_gemm.csv`.

3. Test the performance, modify the test instance in `op_tests/testflatmm.py` and run it, please wait a few minutes as it will build gemm_a8w8_bpreshuffle_cktile kernels in `aiter/configs/a8w8_bpreshuffle_cktile_tuned_gemm.csv` via jit：  
`GEMM_CKTILE_BPRESHUFFLE_HIP_CLANG_PATH=/data/llvm-project/build/bin/ python3 op_tests/testflatmm.py`


## More
If you want to re-install gemm_a8w8_bpreshuffle_cktile, you should remove `aiter/jit/module_gemm_a8w8_bpreshuffle_cktile.so` and `aiter/jit/build/module_gemm_a8w8_bpreshuffle_cktile` first.
If you use flag `PREBUILD_KERNELS=1 USE_CK_A8W8=1` when you install aiter, it will build gemm a8w8 kernels in `aiter/configs/a8w8_bpreshuffle_cktile_tuned_gemm.csv` by default. If you want to use the new result of gemm_a8w8_bpreshuffle_cktile_tune, please remove `build` and `*.so` first, then re-intall aiter after finishing tune.
