# CK gemm a8w8 tune

1. Install aiter:
`cd $aiter_path`
`python3 setup.py develop`

2. Add GEMM shapes in `aiter/configs/a8w8_bpreshuffle_untuned_gemm.csv`
    |**M**|**N**|**K**|******q_dtype_w******|
    |-----|-----|-----|---------------------|
    |128  |1536 |7168 |torch.float8_e4m3fnuz|


3. Start tuning:
Run the following cmd to start tuning, please wait a few minutes as it will build gemm_a8w8_bpreshuffle_tune via jit:
`python3 csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py -i aiter/configs/a8w8_bpreshuffle_untuned_gemm.csv -o aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv`
You can find the results of this tuning in `aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv`.
    |**cu_num**|**M**|**N**|**K**|******q_dtype_w******|**kernelId**|**splitK**|**us**|**kernelName**|
    |----------|-----|-----|-----|---------------------|------------|----------|------|--------------|
    |80        |128  |1536 |7168 |torch.float8_e4m3fnuz|23          |0         |32.99 |xxxxxxxx      |

    `cu_num` means the number of compute units, and it is used to distinguish between graphics.
    `q_dtype_w` means the quantization data type of weight, and it is used to distinguish between different quantization data types. support torch.int8 and fp8

4. Build tuned kernels and test:
Test the performance, modify the test instance in `op_tests/test_gemm_a8w8.py` and run it, please wait a few minutes as it will build gemm_a8w8 tuned kernels in `aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv` via jit:
`python3 op_tests/test_gemm_a8w8.py`
If you have built gemm_a8w8_bpreshuffle kernels brefore tuning new GEMM shapes, please add `AITER_REBUILD=1` before your test cmd, such as `AITER_REBUILD=1 python3 op_tests/test_gemm_a8w8.py`. It will rebuild kernels from `AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE`  the default one will be aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv`.

## More
If you use flag `PREBUILD_KERNELS=1` when you install aiter, it will build gemm a8w8 bpreshuffle kernels in tuned bpreshuffle gemm csv by default. If you want to use the new result of gemm_a8w8_bpreshuffle_tune, please remove `build` and `*.so` in `aiter/jit` first, then re-intall aiter after finishing tune. This can take a lot of time and is not recommended.
