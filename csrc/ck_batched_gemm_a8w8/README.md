# CK batched_gemm a8w8 tune

1. Install aiter:  
`python3 setup.py develop`

2. Tune batched_gemm a8w8: 
 First add batched_gemm shapes in `aiter/configs/a8w8_untuned_batched_gemm.csv`, then run the following cmd to start tuning, please wait a few minutes as it will build batched_gemm_a8w8_tune via jit:  
`python3 csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8_tune.py -i aiter/configs/a8w8_untuned_batched_gemm.csv -o aiter/configs/a8w8_tuned_batched_gemm.csv`  
You can find the results of the tuning in `aiter/configs/a8w8_tuned_batched_gemm.csv`.

3. Test the performance, modify the test instance in `op_tests/test_batched_gemm_a8w8.py` and run it, please wait a few minutes as it will build batched_gemm_a8w8 kernels in `aiter/configs/a8w8_tuned_batched_gemm.csv` via jit：  
`python3 op_tests/test_batched_gemm_a8w8.py`


## More
If you want to re-install batched_gemm_a8w8, you should remove `aiter/jit/module_batched_gemm_a8w8.so` and `aiter/jit/build/module_batched_gemm_a8w8` first.
If you use flag `PREBUILD_KERNELS=1 USE_CK_A8W8=1` when you install aiter, it will build batched_gemm a8w8 kernels in `aiter/configs/a8w8_tuned_batched_gemm.csv` by default. If you want to use the new result of batched_gemm_a8w8_tune, please remove `build` and `*.so` first, then re-intall aiter after finishing tune.
