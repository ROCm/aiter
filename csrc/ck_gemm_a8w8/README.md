# CK gemm a8w8 tune

1. Install aiter:  
`python3 setup.py develop`

2. Tune gemm a8w8: 
 First add GEMM shapes in `aiter/configs/a8w8_untuned_gemm.csv`, then run the following cmd to start tuning, please wait a few minutes as it will build gemm_a8w8_tune via jit:  
`python3 csrc/ck_gemm_a8w8/gemm_a8w8_tune.py -i aiter/configs/a8w8_untuned_gemm.csv -o aiter/configs/a8w8_tuned_gemm.csv`  
If you want to use split K kernels, you can add the `-k` parameter at the end, notice that should change `bias` to `bias/(2^k)`.
You can find the results of the tuning in `aiter/configs/a8w8_tuned_gemm.csv`.

3. Test the performance, modify the test instance in `op_tests/test_gemm_a8w8.py` and run it, please wait a few minutes as it will build gemm_a8w8 kernels in `aiter/configs/a8w8_tuned_gemm.csv` via jit：  
`python3 op_tests/test_gemm_a8w8.py`


## More
If you want to re-install gemm_a8w8, you should remove `aiter/jit/module_gemm_a8w8.so` and `aiter/jit/build/module_gemm_a8w8` first.
If you use flag `PREBUILD_KERNELS=1 USE_CK_A8W8=1` when you install aiter, it will build gemm a8w8 kernels in `aiter/configs/a8w8_tuned_gemm.csv` by default. If you want to use the new result of gemm_a8w8_tune, please remove `build` and `*.so` first, then re-intall aiter after finishing tune.


## FP8 A8W8 Rowwise Scaling GEMM

The following steps will walk you through the full process of getting the best performance out of your hardware.

0. Clear gemm_a8w8: Remove `aiter/jit/module_gemm_a8w8.so` and `aiter/jit/build/module_gemm_a8w8`.
```bash
rm -rf aiter/jit/module_gemm_a8w8.so
rm -rf aiter/jit/build/module_gemm_a8w8
```

1. Install the AITER library
```bash
python3 setup.py develop
```

2. Tune your GEMM kernel

python3 csrc/ck_gemm_a8w8/gemm_a8w8_tune.py -i aiter/configs/a8w8_gemm_model_config/Mixtral-8x7B-Instruct-v0.1-TP1.csv -o aiter/configs/a8w8_tuned_gemm.csv --dtype fp8 -k

3. Use the operator:
```python
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_CK

output = gemm_a8w8_CK(
    qinput, # [M, K]
    weight, # [N, K]
    x_scale, # [M, 1]
    weight_scale, # [1, N]
    bias,
    dtype=out_dtype # torch.bfloat16, torch.float16
)
```
