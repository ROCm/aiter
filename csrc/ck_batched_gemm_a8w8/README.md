# CK batched_gemm a8w8 tune

1. Install aiter:
`cd $aiter_path`
`python3 setup.py develop`

2. Add GEMM shapes in `aiter/configs/a8w8_untuned_batched_gemm.csv`
    |**B**|**M**|**N**|**K**|
    |-----|-----|-----|-----|
    |16   |128  |1536 |7168 |

3. Start tuning:
Run the following cmd to start tuning, run the following cmd to start tuning, please wait a few minutes as it will build batched_gemm_a8w8_tune via jit:
`python3 csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8_tune.py -i aiter/configs/a8w8_untuned_batched_gemm.csv -o aiter/configs/a8w8_tuned_batched_gemm.csv`
You can find the results of the tuning in `aiter/configs/a8w8_tuned_batched_gemm.csv`.

4. Build tuned kernels and test:
Test the performance, modify the test instance in `op_tests/test_batched_gemm_a8w8.py` and run it, please wait a few minutes as it will build batched_gemm_a8w8 tuned kernels in `aiter/configs/a8w8_tuned_batched_gemm.csv` via jit:
`python3 op_tests/test_batched_gemm_a8w8.py`
If you have built batched_gemm_a8w8 kernels brefore tuning new GEMM shapes, please add `AITER_REBUILD=1` before your test cmd, such as `AITER_REBUILD=1 python3 op_tests/test_batched_gemm_a8w8.py`. It will rebuild kernels from `aiter/configs/a8w8_tuned_batched_gemm.csv`.


## More
If you use flag `PREBUILD_KERNELS=1` when you install aiter, it will build gemm a8w8 kernels in tuned gemm csv by default. If you want to use the new result of gemm_a8w8_tune, please remove `build` and `*.so` in `aiter/jit` first, then re-intall aiter after finishing tune. This can take a lot of time and is not recommended.
